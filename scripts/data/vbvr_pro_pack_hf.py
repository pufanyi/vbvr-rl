"""Package a manifest-filtered raw VBVR-Pro dataset for Hugging Face.

The source dataset can contain hundreds of thousands of small files. This
script writes deterministic, size-bounded WebDataset tar shards instead, plus
a dataset card, a sanitized source manifest, per-file checksums, and an audit
report. It never writes into the source tree.

Example:
    .venv/bin/python -m scripts.data.vbvr_pro_pack_hf \
        --dataset-json data/vbvr_pro/vbvr_pro_rl_indomain_256x256x161_evalkit_6fedd9d9.json \
        --output-dir storage/hf/vbvr-pro-rl-indomain-50k \
        --repo-id pufanyi/vbvr-pro-rl-indomain-50k \
        --license-file /mnt/aigc/xujunxiang/Code/VBVR-Pro/LICENSE \
        --expected-samples 50000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from loguru import logger

from src.data.i2v_dataset import I2VDataset, _VBVRProSample

_REQUIRED_FILES = frozenset(
    {
        "first_frame.png",
        "image/prompt.txt",
        "metadata.json",
        "video/final_frame.png",
        "video/ground_truth.mp4",
        "video/prompt.txt",
    }
)
_KNOWN_FIELDS = {
    "first_frame.png": "first.png",
    "metadata.json": "metadata.json.bin",
    "video/final_frame.png": "final.png",
    "video/ground_truth.mp4": "gt.mp4",
    "video/prompt.txt": "video_prompt.txt",
    "image/prompt.txt": "image_prompt.txt",
}
_TEXT_SUFFIXES = frozenset({".json", ".txt"})
_AUDIT_PATTERNS = {
    "internal_mount_path": re.compile(rb"/mnt/(?:umm|aigc)/", re.IGNORECASE),
    "home_path": re.compile(rb"/home/[A-Za-z0-9._-]+/", re.IGNORECASE),
    "internal_username": re.compile(rb"\b(?:pufanyi|xujunxiang)\b", re.IGNORECASE),
    "email_address": re.compile(rb"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "private_key": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "credential_field": re.compile(
        rb'"(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|password|token)"'
        rb'\s*:\s*"(?!none|null|redacted|unset)[^"]{4,}"',
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class _SelectedSample:
    position: int
    sample: _VBVRProSample
    split: str
    task_split: str
    num_frames: int
    height: int | None
    width: int | None
    fps: int

    @property
    def key(self) -> str:
        return f"{self.position:08d}"


@dataclass(frozen=True)
class _SourceFile:
    path: Path
    relative_path: str
    field: str
    size: int


class _Audit:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    def scanner(self, sample_key: str, relative_path: str) -> _FileAudit:
        return _FileAudit(self, sample_key, relative_path)

    def record(self, label: str, sample_key: str, relative_path: str) -> None:
        self.counts[label] += 1
        examples = self.examples[label]
        if len(examples) < 20:
            examples.append({"key": sample_key, "file": relative_path})

    def as_dict(self) -> dict:
        return {
            "clean": not self.counts,
            "finding_file_counts": dict(sorted(self.counts.items())),
            "examples": dict(sorted(self.examples.items())),
            "note": "Values are intentionally omitted; counts are per source file and pattern.",
        }


class _FileAudit:
    def __init__(self, audit: _Audit, sample_key: str, relative_path: str) -> None:
        self.audit = audit
        self.sample_key = sample_key
        self.relative_path = relative_path
        self.tail = b""
        self.seen: set[str] = set()

    def feed(self, data: bytes) -> None:
        combined = self.tail + data
        for label, pattern in _AUDIT_PATTERNS.items():
            if label not in self.seen and pattern.search(combined):
                self.seen.add(label)
                self.audit.record(label, self.sample_key, self.relative_path)
        self.tail = combined[-512:]


class _HashingReader:
    def __init__(self, raw: BinaryIO, scanner: _FileAudit | None) -> None:
        self.raw = raw
        self.scanner = scanner
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        if data:
            self.digest.update(data)
            if self.scanner is not None:
                self.scanner.feed(data)
        return data

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


class _HashingWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = str(path)
        self.raw = path.open("wb")
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return self.raw.write(data)

    def tell(self) -> int:
        return self.raw.tell()

    def flush_and_close(self) -> None:
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


class _ShardWriter:
    def __init__(self, output_dir: Path, target_bytes: int, audit: _Audit) -> None:
        self.output_dir = output_dir
        self.target_bytes = target_bytes
        self.audit = audit
        self.shard_id = 0
        self.tar: tarfile.TarFile | None = None
        self.hashing_writer: _HashingWriter | None = None
        self.tmp_path: Path | None = None
        self.final_path: Path | None = None
        self.estimated_bytes = 0
        self.samples_in_shard = 0
        self.completed: list[dict] = []

    @staticmethod
    def _entry_bytes(size: int) -> int:
        return 512 + ((size + 511) // 512) * 512

    def _open(self) -> None:
        self.final_path = self.output_dir / f"shard-{self.shard_id:05d}.tar"
        self.tmp_path = self.output_dir / f".shard-{self.shard_id:05d}.tar.tmp"
        self.hashing_writer = _HashingWriter(self.tmp_path)
        self.tar = tarfile.open(  # noqa: SIM115
            fileobj=self.hashing_writer,
            mode="w|",
            format=tarfile.PAX_FORMAT,
        )
        self.estimated_bytes = 0
        self.samples_in_shard = 0

    def _close(self) -> None:
        if self.tar is None:
            return
        assert self.hashing_writer is not None
        assert self.tmp_path is not None and self.final_path is not None
        self.tar.close()
        self.hashing_writer.flush_and_close()
        self.tmp_path.replace(self.final_path)
        size = self.final_path.stat().st_size
        record = {
            "name": self.final_path.name,
            "size": size,
            "sha256": self.hashing_writer.hexdigest(),
            "samples": self.samples_in_shard,
        }
        self.completed.append(record)
        logger.info(
            "closed {}: {} samples, {:.3f} GiB, sha256={}",
            self.final_path.name,
            self.samples_in_shard,
            size / 2**30,
            record["sha256"][:16],
        )
        self.shard_id += 1
        self.tar = None
        self.hashing_writer = None
        self.tmp_path = None
        self.final_path = None

    @staticmethod
    def _normalized_info(name: str, size: int) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name)
        info.size = size
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        return info

    def _add_bytes(self, name: str, data: bytes) -> str:
        assert self.tar is not None
        self.tar.addfile(self._normalized_info(name, len(data)), BytesIO(data))
        return hashlib.sha256(data).hexdigest()

    def _add_source(self, key: str, source: _SourceFile) -> str:
        assert self.tar is not None
        scanner = (
            self.audit.scanner(key, source.relative_path) if source.path.suffix.lower() in _TEXT_SUFFIXES else None
        )
        with source.path.open("rb") as raw:
            reader = _HashingReader(raw, scanner)
            self.tar.addfile(self._normalized_info(f"{key}.{source.field}", source.size), reader)
            if raw.read(1):
                raise OSError(f"source changed size while packing: {source.path}")
        return reader.hexdigest()

    def _build_extras(self, key: str, extras: list[_SourceFile]) -> tuple[bytes, list[dict]]:
        buffer = BytesIO()
        records = []
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for source in extras:
                scanner = (
                    self.audit.scanner(key, source.relative_path)
                    if source.path.suffix.lower() in _TEXT_SUFFIXES
                    else None
                )
                with source.path.open("rb") as raw:
                    reader = _HashingReader(raw, scanner)
                    chunks = []
                    while chunk := reader.read(8 * 1024 * 1024):
                        chunks.append(chunk)
                    data = b"".join(chunks)
                if len(data) != source.size:
                    raise OSError(f"source changed size while packing: {source.path}")
                info = zipfile.ZipInfo(source.relative_path, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
                records.append(
                    {
                        "path": source.relative_path,
                        "field": f"extras.zip.bin::{source.relative_path}",
                        "size": source.size,
                        "sha256": reader.hexdigest(),
                    }
                )
        return buffer.getvalue(), records

    def write(
        self,
        selected: _SelectedSample,
        files: list[_SourceFile],
        sample_json: bytes,
    ) -> tuple[str, list[dict]]:
        direct = [source for source in files if source.relative_path in _KNOWN_FIELDS]
        extras = [source for source in files if source.relative_path not in _KNOWN_FIELDS]
        estimate = (
            self._entry_bytes(len(sample_json))
            + sum(self._entry_bytes(source.size) for source in direct)
            + self._entry_bytes(sum(source.size for source in extras) + 4096)
        )
        if self.tar is not None and self.samples_in_shard and self.estimated_bytes + estimate > self.target_bytes:
            self._close()
        if self.tar is None:
            self._open()
        assert self.final_path is not None
        shard_name = self.final_path.name

        self._add_bytes(f"{selected.key}.json", sample_json)
        file_records = []
        for source in direct:
            digest = self._add_source(selected.key, source)
            file_records.append(
                {
                    "path": source.relative_path,
                    "field": source.field,
                    "size": source.size,
                    "sha256": digest,
                }
            )
        extras_blob, extras_records = self._build_extras(selected.key, extras)
        self._add_bytes(f"{selected.key}.extras.zip.bin", extras_blob)
        file_records.extend(extras_records)
        self.estimated_bytes += estimate
        self.samples_in_shard += 1
        return shard_name, file_records

    def close(self) -> None:
        self._close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | os.PathLike[str], parent: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _load_selected_samples(dataset_json: Path) -> tuple[list[_SelectedSample], list[dict], dict]:
    raw = json.loads(dataset_json.read_text(encoding="utf-8"))
    entries = [raw] if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"dataset JSON must contain one or more entries: {dataset_json}")

    selected: list[_SelectedSample] = []
    sanitized_tasks: list[dict] = []
    task_seen: set[tuple[str, str]] = set()
    common_config: dict | None = None
    for entry in entries:
        fmt = str(entry.get("format", entry.get("type", "parquet"))).lower().replace("-", "_")
        if fmt != "vbvr_pro":
            raise ValueError(f"only format=vbvr_pro is supported, got {fmt!r}")
        split = str(entry.get("split", "train"))
        manifest_path = _resolve_path(entry["split_manifest"], dataset_json.parent)
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_by_task = {
            str(record.get("task") or Path(str(record["source"])).parent.name): record for record in records
        }
        samples = I2VDataset._load_vbvr_pro_manifest(entry, dataset_json.parent)
        cfg = {
            "num_frames": int(entry.get("num_frames", 81)),
            "height": entry.get("height"),
            "width": entry.get("width"),
            "fps": int(entry.get("fps", 16)),
        }
        if common_config is None:
            common_config = cfg
        elif common_config != cfg:
            raise ValueError("all dataset entries must use the same frame/size/fps config")

        selected_ids: dict[str, list[str]] = defaultdict(list)
        for sample in samples:
            record = record_by_task[sample.task_name]
            task_split = str(record.get("split", ""))
            selected.append(
                _SelectedSample(
                    position=len(selected),
                    sample=sample,
                    split=split,
                    task_split=task_split,
                    num_frames=cfg["num_frames"],
                    height=cfg["height"],
                    width=cfg["width"],
                    fps=cfg["fps"],
                )
            )
            selected_ids[sample.task_name].append(sample.sample_id)

        for task_name, sample_ids in selected_ids.items():
            key = (task_name, split)
            if key in task_seen:
                raise ValueError(f"duplicate task/split across entries: {key}")
            task_seen.add(key)
            record = record_by_task[task_name]
            sanitized_tasks.append(
                {
                    "task": task_name,
                    "task_split": str(record.get("split", "")),
                    "selected_split": split,
                    "sample_count": len(sample_ids),
                    "sample_ids": sample_ids,
                }
            )

    assert common_config is not None
    return selected, sanitized_tasks, common_config


def _field_name(relative_path: str, used: set[str]) -> str:
    field = _KNOWN_FIELDS.get(relative_path)
    if field is None and relative_path.startswith("image/"):
        field = "image_" + relative_path.removeprefix("image/")
    if field is None:
        field = "source_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", relative_path)
    if "/" in field or not field:
        raise ValueError(f"invalid WebDataset field for {relative_path!r}: {field!r}")
    if field in used:
        stem, dot, suffix = field.partition(".")
        field = f"{stem}_{hashlib.sha256(relative_path.encode()).hexdigest()[:12]}{dot}{suffix}"
    if field in used:
        raise ValueError(f"WebDataset field collision for {relative_path!r}: {field!r}")
    used.add(field)
    return field


def _discover_files(sample: _VBVRProSample) -> list[_SourceFile]:
    if not sample.sample_dir.is_dir():
        raise FileNotFoundError(f"missing sample directory: {sample.sample_dir}")
    files: list[_SourceFile] = []
    used: set[str] = set()
    observed: set[str] = set()
    for path in sorted(sample.sample_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing to package symlink: {path}")
        if not path.is_file():
            continue
        relative_path = path.relative_to(sample.sample_dir).as_posix()
        observed.add(relative_path)
        files.append(
            _SourceFile(
                path=path,
                relative_path=relative_path,
                field=_field_name(relative_path, used),
                size=path.stat().st_size,
            )
        )
    missing = sorted(_REQUIRED_FILES - observed)
    if missing:
        raise FileNotFoundError(f"{sample.sample_dir} is missing required files: {missing}")
    return files


def _sample_record(selected: _SelectedSample, files: list[_SourceFile]) -> dict:
    prompt_path = selected.sample.sample_dir / "video" / "prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    return {
        "key": selected.key,
        "manifest_position": selected.position,
        "task_name": selected.sample.task_name,
        "sample_id": selected.sample.sample_id,
        "split": selected.split,
        "task_split": selected.task_split,
        "prompt": prompt,
        "num_frames": selected.num_frames,
        "height": selected.height,
        "width": selected.width,
        "fps_metadata": selected.fps,
        "source_files": {
            (
                source.field if source.relative_path in _KNOWN_FIELDS else f"extras.zip.bin::{source.relative_path}"
            ): source.relative_path
            for source in files
        },
    }


def _dataset_card(
    *,
    repo_id: str,
    sample_count: int,
    task_count: int,
    shard_count: int,
    source_bytes: int,
    archive_bytes: int,
    config: dict,
    manifest_sha256: str,
    dataset_json_sha256: str,
    shuffle_seed: int | None,
) -> str:
    shuffle_text = (
        f"Samples were deterministically shuffled across shards with seed `{shuffle_seed}`."
        if shuffle_seed is not None
        else "Samples retain manifest order."
    )
    return f"""\
---
license: apache-2.0
tags:
- video
- image-to-video
- visual-reasoning
- reinforcement-learning
- webdataset
pretty_name: VBVR-Pro RL In-Domain 50K
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/shard-*.tar
---

# VBVR-Pro RL In-Domain 50K

This repository contains the exact `{sample_count:,}` samples selected by the
Wan-Trainer raw-data descriptor
`vbvr_pro_rl_indomain_256x256x161_evalkit_6fedd9d9.json`: `{task_count}` EvalKit
In-Domain tasks with 1,000 RL samples per task.

The original small-file tree was losslessly repackaged into `{shard_count}`
WebDataset tar shards. All files inside every selected sample directory are
included. Internal source filesystem paths are not included in the published
manifests.

## Dataset facts

- Samples: **{sample_count:,}**
- Tasks: **{task_count}**
- Source file bytes: **{source_bytes / 1e9:.3f} GB**
- Tar archive bytes: **{archive_bytes / 1e9:.3f} GB**
- Training view: `{config["width"]}x{config["height"]}`, `{config["num_frames"]}` frames
- Source `fps` field: `{config["fps"]}` (metadata only in the current Wan-Trainer raw loader)
- Source split-manifest SHA-256: `{manifest_sha256}`
- Dataset descriptor SHA-256: `{dataset_json_sha256}`

{shuffle_text}

## WebDataset fields

Each sample uses an eight-digit key such as `00001234`.

| Field | Content |
|---|---|
| `.json` | Published sample index: task, sample ID, prompt, dimensions, and field mapping |
| `.metadata.json.bin` | Original VBVR-Pro declarative-render JSON bytes |
| `.gt.mp4` | Original ground-truth video |
| `.first.png` | Original I2V conditioning frame |
| `.final.png` | Original final frame |
| `.video_prompt.txt` | Original video prompt |
| `.image_prompt.txt` | Original image prompt |
| `.extras.zip.bin` | Deterministic ZIP containing original image-sequence frames and any extra files |

`samples.jsonl` contains the shard assignment and SHA-256 of every original
source file. `source_manifest.json` is the selected manifest with internal
absolute source paths removed. `SHA256SUMS` covers the published artifacts.

## Streaming

```python
import json
import webdataset as wds

url = "hf://datasets/{repo_id}/data/shard-{{00000..{shard_count - 1:05d}}}.tar"
dataset = wds.WebDataset(url, shardshuffle=True).shuffle(1000).decode()

for sample in dataset:
    index = sample["json"]
    metadata = json.loads(sample["metadata.json.bin"])
    video_bytes = sample["gt.mp4"]
    first_frame = sample["first.png"]
    break
```

The `metadata.json.bin` field can be large for frame-explicit tasks. Select only
the fields needed by your pipeline when possible.

## License and provenance

The upstream VBVR-Pro repository carries the Apache License 2.0; its license
text is reproduced in this dataset repository. The source split contains only
synthetically generated visual-reasoning samples.

The `In-Domain` label is relative to the VBVR EvalKit task taxonomy. These
tasks are not held out from the corresponding RL training distribution.
"""


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _artifact_checksums(output_dir: Path, known_sha256: dict[str, str] | None = None) -> None:
    known_sha256 = known_sha256 or {}
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and not path.name.endswith(".tmp")
    )
    lines = []
    for path in paths:
        relative = path.relative_to(output_dir).as_posix()
        digest = known_sha256.get(relative) or _sha256(path)
        lines.append(f"{digest}  {relative}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--license-file", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--target-shard-gib", type=float, default=1.0)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_json = args.dataset_json.resolve()
    output_dir = args.output_dir.resolve()
    license_file = args.license_file.resolve()
    if not dataset_json.is_file():
        raise FileNotFoundError(dataset_json)
    if not license_file.is_file():
        raise FileNotFoundError(license_file)
    if args.target_shard_gib <= 0:
        raise ValueError("--target-shard-gib must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    selected, sanitized_tasks, config = _load_selected_samples(dataset_json)
    if args.expected_samples is not None and len(selected) != args.expected_samples:
        raise ValueError(f"expected {args.expected_samples} samples, selected {len(selected)}")
    pack_order = list(selected)
    shuffle_seed = None if args.no_shuffle else args.shuffle_seed
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(pack_order)
    if args.max_samples is not None:
        pack_order = pack_order[: args.max_samples]
        selected = list(pack_order)
        selected_ids = {(item.sample.task_name, item.sample.sample_id) for item in selected}
        sanitized_tasks = [
            {
                **task,
                "sample_ids": [
                    sample_id for sample_id in task["sample_ids"] if (task["task"], sample_id) in selected_ids
                ],
            }
            for task in sanitized_tasks
        ]
        sanitized_tasks = [
            {**task, "sample_count": len(task["sample_ids"])} for task in sanitized_tasks if task["sample_ids"]
        ]

    audit = _Audit()
    writer = _ShardWriter(output_dir / "data", int(args.target_shard_gib * 2**30), audit)
    source_bytes = 0
    source_files = 0
    task_counts: Counter[str] = Counter()
    samples_index_path = output_dir / "samples.jsonl"
    try:
        with samples_index_path.open("w", encoding="utf-8") as index_stream:
            for packed_position, item in enumerate(pack_order, 1):
                files = _discover_files(item.sample)
                record = _sample_record(item, files)
                sample_json = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                shard_name, file_records = writer.write(item, files, sample_json)
                index_stream.write(
                    json.dumps(
                        {
                            **record,
                            "shard": f"data/{shard_name}",
                            "files": file_records,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                source_bytes += sum(source.size for source in files)
                source_files += len(files)
                task_counts[item.sample.task_name] += 1
                if packed_position % 1000 == 0 or packed_position == len(pack_order):
                    logger.info(
                        "packed {}/{} samples ({:.3f} GiB source)",
                        packed_position,
                        len(pack_order),
                        source_bytes / 2**30,
                    )
    finally:
        writer.close()

    raw_entries = json.loads(dataset_json.read_text(encoding="utf-8"))
    entries = [raw_entries] if isinstance(raw_entries, dict) else raw_entries
    manifest_paths = {_resolve_path(entry["split_manifest"], dataset_json.parent) for entry in entries}
    manifest_hashes = {_sha256(path) for path in manifest_paths}
    manifest_sha256 = ",".join(sorted(manifest_hashes))
    archive_bytes = sum(record["size"] for record in writer.completed)

    _write_json(
        output_dir / "dataset_config.json",
        {
            "repo_id": args.repo_id,
            "samples": len(selected),
            "tasks": len(task_counts),
            "source_files": source_files,
            "source_bytes": source_bytes,
            "archive_bytes": archive_bytes,
            "shards": writer.completed,
            "training_view": config,
            "shuffle_seed": shuffle_seed,
            "dataset_json_sha256": _sha256(dataset_json),
            "source_manifest_sha256": sorted(manifest_hashes),
        },
    )
    _write_json(output_dir / "source_manifest.json", sanitized_tasks)
    _write_json(output_dir / "audit.json", audit.as_dict())
    shutil.copyfile(license_file, output_dir / "LICENSE")
    (output_dir / "README.md").write_text(
        _dataset_card(
            repo_id=args.repo_id,
            sample_count=len(selected),
            task_count=len(task_counts),
            shard_count=len(writer.completed),
            source_bytes=source_bytes,
            archive_bytes=archive_bytes,
            config=config,
            manifest_sha256=manifest_sha256,
            dataset_json_sha256=_sha256(dataset_json),
            shuffle_seed=shuffle_seed,
        ),
        encoding="utf-8",
    )
    _artifact_checksums(
        output_dir,
        {f"data/{record['name']}": record["sha256"] for record in writer.completed},
    )

    logger.info(
        "complete: {} samples, {} tasks, {} source files, {:.3f} GiB in {} shards",
        len(selected),
        len(task_counts),
        source_files,
        archive_bytes / 2**30,
        len(writer.completed),
    )
    if audit.counts:
        raise RuntimeError(f"audit findings require review before upload: {dict(audit.counts)}")


if __name__ == "__main__":
    main()
