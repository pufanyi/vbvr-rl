"""Materialize the published raw VBVR-Pro WebDataset into an I2VDataset tree.

The public ``pufanyi/vbvr-pro-rl-indomain-50k`` snapshot is a lossless raw
backup, not a latent WebDataset.  ``I2VDataset`` and ``vbvr_rule`` need stable
filesystem paths for the source video, first/final frames, and metadata.  This
utility restores only those training/reward-critical fields and writes the
standard VBVR-Pro descriptor consumed by ``I2VDataset``.

Example:
    .venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
        --dataset-root storage/datasets/vbvr-pro-rl-indomain-50k \
        --output-dir storage/datasets/vbvr-pro-rl-indomain-50k/materialized \
        --expected-samples 50000 \
        --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

_FIELD_TARGETS = {
    "first.png": Path("first_frame.png"),
    "metadata.json.bin": Path("metadata.json"),
    "final.png": Path("video/final_frame.png"),
    "gt.mp4": Path("video/ground_truth.mp4"),
    "video_prompt.txt": Path("video/prompt.txt"),
}
_FIELD_SUFFIXES = tuple(sorted([(f".{field}", field) for field in _FIELD_TARGETS], reverse=True))


@dataclass(frozen=True)
class _ExpectedFile:
    size: int
    sha256: str


@dataclass(frozen=True)
class _PackedSample:
    key: str
    task_name: str
    sample_id: str
    task_split: str
    shard: Path
    files: dict[str, _ExpectedFile]


@dataclass(frozen=True)
class _ShardResult:
    shard: Path
    samples: int
    written_files: int
    reused_files: int
    written_bytes: int


def _safe_component(value: str, *, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _load_samples(
    dataset_root: Path,
    *,
    max_samples: int | None,
) -> list[_PackedSample]:
    index_path = dataset_root / "samples.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    samples: list[_PackedSample] = []
    keys: set[str] = set()
    with index_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            record = json.loads(line)
            key = _safe_component(str(record["key"]), label=f"key at line {line_number}")
            if key in keys:
                raise ValueError(f"duplicate sample key in {index_path}: {key}")
            keys.add(key)
            task_name = _safe_component(str(record["task_name"]), label=f"task_name for {key}")
            sample_id = _safe_component(str(record["sample_id"]), label=f"sample_id for {key}")
            shard = dataset_root / str(record["shard"])
            if not shard.is_file():
                raise FileNotFoundError(shard)

            expected: dict[str, _ExpectedFile] = {}
            for file_record in record["files"]:
                field = str(file_record["field"])
                if field in _FIELD_TARGETS:
                    expected[field] = _ExpectedFile(
                        size=int(file_record["size"]),
                        sha256=str(file_record["sha256"]),
                    )
            missing = sorted(set(_FIELD_TARGETS) - set(expected))
            if missing:
                raise ValueError(f"sample {key} is missing required fields: {missing}")
            samples.append(
                _PackedSample(
                    key=key,
                    task_name=task_name,
                    sample_id=sample_id,
                    task_split=str(record["task_split"]),
                    shard=shard,
                    files=expected,
                )
            )

    if not samples:
        raise ValueError(f"no samples selected from {index_path}")
    return samples


def _target_path(output_dir: Path, sample: _PackedSample, field: str) -> Path:
    return output_dir / "raw" / sample.task_name / sample.sample_id / _FIELD_TARGETS[field]


def _copy_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
    expected: _ExpectedFile,
    *,
    verify_existing: bool,
) -> tuple[bool, int]:
    if member.size != expected.size:
        raise ValueError(f"archive size mismatch for {member.name}: expected {expected.size}, found {member.size}")
    if (
        target.is_file()
        and target.stat().st_size == expected.size
        and (not verify_existing or _sha256(target) == expected.sha256)
    ):
        return False, 0

    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"could not read tar member: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.pid{os.getpid()}.thread{threading.get_ident()}.tmp")
    digest = hashlib.sha256()
    written = 0
    try:
        with temporary.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != expected.size:
            raise ValueError(f"short read for {member.name}: expected {expected.size}, wrote {written}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected.sha256:
            raise ValueError(f"SHA-256 mismatch for {member.name}: expected {expected.sha256}, found {actual_sha256}")
        os.replace(temporary, target)
    finally:
        source.close()
        temporary.unlink(missing_ok=True)
    return True, written


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _member_field(name: str) -> tuple[str, str] | None:
    if Path(name).name != name:
        raise ValueError(f"unexpected nested WebDataset member: {name!r}")
    for suffix, field in _FIELD_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], field
    return None


def _unpack_shard(
    shard: Path,
    samples: list[_PackedSample],
    output_dir: Path,
    *,
    verify_existing: bool,
) -> _ShardResult:
    by_key = {sample.key: sample for sample in samples}
    seen: dict[str, set[str]] = defaultdict(set)
    written_files = 0
    reused_files = 0
    written_bytes = 0
    with tarfile.open(shard, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parsed = _member_field(member.name)
            if parsed is None:
                continue
            key, field = parsed
            sample = by_key.get(key)
            if sample is None:
                continue
            if field in seen[key]:
                raise ValueError(f"duplicate field {field!r} for sample {key} in {shard}")
            seen[key].add(field)
            written, byte_count = _copy_member(
                archive,
                member,
                _target_path(output_dir, sample, field),
                sample.files[field],
                verify_existing=verify_existing,
            )
            if written:
                written_files += 1
                written_bytes += byte_count
            else:
                reused_files += 1

    missing = {
        sample.key: sorted(set(_FIELD_TARGETS) - seen[sample.key])
        for sample in samples
        if set(_FIELD_TARGETS) - seen[sample.key]
    }
    if missing:
        preview = list(missing.items())[:5]
        raise ValueError(f"{shard} is missing selected sample fields: {preview}")
    return _ShardResult(
        shard=shard,
        samples=len(samples),
        written_files=written_files,
        reused_files=reused_files,
        written_bytes=written_bytes,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pid{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_training_metadata(output_dir: Path, dataset_root: Path, samples: list[_PackedSample]) -> None:
    tasks: OrderedDict[str, dict] = OrderedDict()
    for sample in samples:
        task = tasks.setdefault(
            sample.task_name,
            {
                "task": sample.task_name,
                "source": sample.task_name,
                "split": sample.task_split,
                "rl": [],
            },
        )
        if task["split"] != sample.task_split:
            raise ValueError(f"inconsistent task split for {sample.task_name}")
        task["rl"].append(sample.sample_id)

    manifest_path = output_dir / "split_manifest_rl.json"
    descriptor_path = output_dir / "dataset.json"
    _write_json(manifest_path, list(tasks.values()))
    _write_json(
        descriptor_path,
        [
            {
                "format": "vbvr_pro",
                "split_manifest": manifest_path.name,
                "data_roots": ["raw"],
                "split": "rl",
                "allowed_task_splits": ["In-Domain_50"],
                "check_files": False,
                "num_frames": 161,
                "height": 256,
                "width": 256,
                "fps": 16,
            }
        ],
    )
    _write_json(
        output_dir / "materialization.json",
        {
            "source_dataset_root": os.path.relpath(dataset_root, output_dir),
            "samples": len(samples),
            "tasks": len(tasks),
            "fields": sorted(_FIELD_TARGETS),
            "descriptor": descriptor_path.name,
            "split_manifest": manifest_path.name,
        },
    )


def materialize(
    dataset_root: Path,
    output_dir: Path,
    *,
    expected_samples: int | None = None,
    max_samples: int | None = None,
    workers: int = 4,
    verify_existing: bool = False,
) -> list[_ShardResult]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    samples = _load_samples(dataset_root, max_samples=max_samples)
    if expected_samples is not None and len(samples) != expected_samples:
        raise ValueError(f"expected {expected_samples} samples, selected {len(samples)}")

    by_shard: dict[Path, list[_PackedSample]] = defaultdict(list)
    for sample in samples:
        by_shard[sample.shard].append(sample)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Materializing {} samples from {} shards into {} with {} workers",
        len(samples),
        len(by_shard),
        output_dir,
        workers,
    )
    results: list[_ShardResult] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(by_shard))) as executor:
        futures = {
            executor.submit(
                _unpack_shard,
                shard,
                shard_samples,
                output_dir,
                verify_existing=verify_existing,
            ): shard
            for shard, shard_samples in sorted(by_shard.items())
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.info(
                "Completed {}: {} samples, {} files written, {} reused, {:.2f} GiB written",
                result.shard.name,
                result.samples,
                result.written_files,
                result.reused_files,
                result.written_bytes / 2**30,
            )

    _write_training_metadata(output_dir, dataset_root, samples)
    results.sort(key=lambda result: result.shard)
    logger.info(
        "Materialization complete: {} samples, {} files written, {} reused, {:.2f} GiB written",
        len(samples),
        sum(result.written_files for result in results),
        sum(result.reused_files for result in results),
        sum(result.written_bytes for result in results) / 2**30,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Hash existing same-size files before reusing them (newly extracted files are always verified).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize(
        args.dataset_root,
        args.output_dir,
        expected_samples=args.expected_samples,
        max_samples=args.max_samples,
        workers=args.workers,
        verify_existing=args.verify_existing,
    )


if __name__ == "__main__":
    main()
