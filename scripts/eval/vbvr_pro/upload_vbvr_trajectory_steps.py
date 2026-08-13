#!/usr/bin/env python3
"""Publish native VBVR-Pro trajectory-step MP4s to split HF Datasets.

The complete archive has 180,000 small MP4s. It is split by model so each
Git-backed Dataset stays below Hugging Face's recommended 100,000 files per
repository. Uploads are deterministic, additive, and safe to resume: paths
already present on the Hub are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.vbvr_pro.build_vbvr_trajectory_space import (  # noqa: E402
    DATASET_TEMPLATE_ROOT,
    DEFAULT_TRAJECTORY_ROOT,
    SAMPLERS,
    STEP_COUNT,
    discover_cells,
    sample_paths,
)

DEFAULT_INDEX = REPO_ROOT / "storage/hf_spaces/vbvrpro_sampler_trajectories/data/index.json"
DEFAULT_REPOS = {
    "baseline": "pufanyi/vbvrpro_sampler_trajectories-baseline-steps",
    "checkpoint-2200": "pufanyi/vbvrpro_sampler_trajectories-2200-steps",
}


@dataclass(frozen=True)
class UploadItem:
    source: Path
    remote_path: str


def model_mapping(values: list[str], *, defaults: dict[str, str]) -> dict[str, str]:
    mapping = dict(defaults)
    for value in values:
        model_id, separator, repo_id = value.partition("=")
        if not separator or model_id not in defaults or not repo_id:
            raise ValueError(f"--repo expects MODEL=REPO with MODEL in {sorted(defaults)}, got {value!r}")
        mapping[model_id] = repo_id
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--repo", action="append", default=[], metavar="MODEL=REPO")
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(DEFAULT_REPOS),
        help="Model archive to publish; repeatable. Defaults to both models.",
    )
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Stop after this many new media commits per model; useful for a resumability smoke test.",
    )
    parser.add_argument("--private", action="store_true", help="Create new Dataset repositories as private.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.repos = model_mapping(args.repo, defaults=DEFAULT_REPOS)
    except ValueError as error:
        parser.error(str(error))
    args.models = args.model or list(DEFAULT_REPOS)
    if args.batch_size <= 0 or args.num_threads <= 0 or args.max_retries <= 0:
        parser.error("batch size, thread count, and retry count must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    return args


def build_upload_items(
    trajectory_root: Path,
    *,
    model_id: str,
    expected_samples: int,
) -> list[UploadItem]:
    cells = discover_cells(trajectory_root.resolve(), expected_samples=expected_samples)
    selected = [cell for cell in cells if cell.model.id == model_id]
    if len(selected) != len(SAMPLERS):
        raise ValueError(f"Expected {len(SAMPLERS)} cells for {model_id}, found {len(selected)}")
    items: list[UploadItem] = []
    for cell in selected:
        for relative in sample_paths(cell, expected_samples=expected_samples):
            for step_index in range(STEP_COUNT):
                filename = f"step_{step_index:02d}.mp4"
                items.append(
                    UploadItem(
                        source=cell.source / relative / filename,
                        remote_path=f"videos/{cell.id}/{relative.as_posix()}/{filename}",
                    )
                )
    expected = len(SAMPLERS) * expected_samples * STEP_COUNT
    if len(items) != expected or len({item.remote_path for item in items}) != expected:
        raise ValueError(f"Expected {expected} unique upload items for {model_id}, found {len(items)}")
    return items


def chunks(items: list[UploadItem], size: int) -> list[list[UploadItem]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def upload_metadata(api: HfApi, *, repo_id: str, index_path: Path) -> None:
    metadata = (
        (DATASET_TEMPLATE_ROOT / ".gitattributes", ".gitattributes"),
        (DATASET_TEMPLATE_ROOT / "README.md", "README.md"),
        (index_path, "data/index.json"),
    )
    for source, _ in metadata:
        if not source.is_file():
            raise FileNotFoundError(source)
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add trajectory archive metadata",
        operations=[
            CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=source) for source, remote_path in metadata
        ],
    )


def present_paths(api: HfApi, *, repo_id: str, paths: list[str]) -> set[str]:
    return {
        item.rfilename
        for item in api.get_paths_info(
            repo_id=repo_id,
            repo_type="dataset",
            paths=paths,
            expand=False,
        )
    }


def upload_batch_with_retry(
    api: HfApi,
    *,
    repo_id: str,
    model_id: str,
    batch: list[UploadItem],
    batch_number: int,
    batch_total: int,
    num_threads: int,
    max_retries: int,
) -> None:
    remaining = batch
    for attempt in range(1, max_retries + 1):
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {model_id} native steps {batch_number}/{batch_total}",
                operations=[
                    CommitOperationAdd(path_in_repo=item.remote_path, path_or_fileobj=item.source) for item in remaining
                ],
                num_threads=num_threads,
            )
            return
        except Exception:
            uploaded = present_paths(
                api,
                repo_id=repo_id,
                paths=[item.remote_path for item in remaining],
            )
            remaining = [item for item in remaining if item.remote_path not in uploaded]
            if not remaining:
                print(f"[recover] batch {batch_number}/{batch_total} completed despite client error", flush=True)
                return
            if attempt == max_retries:
                raise
            delay = min(30, 2**attempt)
            print(
                f"[retry] batch {batch_number}/{batch_total}: {len(remaining)} paths remain; "
                f"attempt {attempt + 1}/{max_retries} in {delay}s",
                flush=True,
            )
            time.sleep(delay)


def publish_model(
    api: HfApi,
    *,
    trajectory_root: Path,
    index_path: Path,
    model_id: str,
    repo_id: str,
    expected_samples: int,
    batch_size: int,
    num_threads: int,
    max_retries: int,
    max_batches: int | None,
    private: bool,
    dry_run: bool,
) -> None:
    items = build_upload_items(
        trajectory_root,
        model_id=model_id,
        expected_samples=expected_samples,
    )
    if dry_run:
        print(f"[dry-run] {model_id}: {len(items)} videos -> {repo_id}", flush=True)
        return

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    upload_metadata(api, repo_id=repo_id, index_path=index_path)
    existing = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    pending = [item for item in items if item.remote_path not in existing]
    print(
        f"[resume] {model_id}: total={len(items)} existing={len(items) - len(pending)} "
        f"pending={len(pending)} repo={repo_id}",
        flush=True,
    )
    batches = chunks(pending, batch_size)
    if max_batches is not None:
        batches = batches[:max_batches]
    for batch_number, batch in enumerate(batches, start=1):
        upload_batch_with_retry(
            api,
            repo_id=repo_id,
            model_id=model_id,
            batch=batch,
            batch_number=batch_number,
            batch_total=len(batches),
            num_threads=num_threads,
            max_retries=max_retries,
        )
        completed = min(batch_number * batch_size, len(pending))
        uploaded_bytes = sum(item.source.stat().st_size for item in batch)
        print(
            f"[upload] {model_id} {batch_number}/{len(batches)}: "
            f"+{len(batch)} files / {uploaded_bytes / 1024**2:.1f} MiB; "
            f"this_run={completed}/{len(pending)}",
            flush=True,
        )

    if max_batches is None:
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        expected_paths = {item.remote_path for item in items}
        missing = sorted(expected_paths - remote_files)
        unexpected_media = sorted(
            path for path in remote_files - expected_paths if path.startswith("videos/") and path.endswith(".mp4")
        )
        if missing or unexpected_media:
            raise RuntimeError(
                f"{repo_id}: missing={len(missing)}, unexpected_media={len(unexpected_media)}, "
                f"first_missing={missing[:1]}, first_unexpected={unexpected_media[:1]}"
            )
        print(
            f"[complete] {model_id}: {len(items)} native step videos are present; repository_files={len(remote_files)}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    api = HfApi()
    for model_id in args.models:
        publish_model(
            api,
            trajectory_root=args.trajectory_root,
            index_path=args.index,
            model_id=model_id,
            repo_id=args.repos[model_id],
            expected_samples=args.expected_samples,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            max_retries=args.max_retries,
            max_batches=args.max_batches,
            private=args.private,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
