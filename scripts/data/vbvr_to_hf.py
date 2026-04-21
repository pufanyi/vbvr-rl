"""Convert VBVR-Bench GT directory to a standard HuggingFace parquet dataset.

Reads:
    <gt_base>/{In-Domain_50, Out-of-Domain_50}/<task>/<idx>/
        ├── first_frame.png
        ├── final_frame.png
        ├── ground_truth.mp4
        └── prompt.txt

Writes two splits (`in_domain`, `out_of_domain`) with schema:
    task_name, video_idx, domain, prompt, first_frame (Image),
    final_frame (Image), ground_truth_video (MP4 bytes).

Deps:
    uv add datasets huggingface_hub

Usage:
    # Local convert only (Arrow on disk)
    uv run python scripts/data/vbvr_to_hf.py \\
        --gt_base /mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench \\
        --output_dir storage/datasets/vbvr_bench_hf

    # Convert + push to Hub (requires `hf auth login` or HF_TOKEN env)
    uv run python scripts/data/vbvr_to_hf.py \\
        --gt_base /mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench \\
        --push_to_hub pufanyi/VBVR-Bench
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi
from loguru import logger

_DOMAIN_DIRS: dict[str, str] = {
    "In-Domain_50": "in_domain",
    "Out-of-Domain_50": "out_of_domain",
}

_FEATURES = Features(
    {
        "task_name": Value("string"),
        "video_idx": Value("string"),
        "domain": Value("string"),
        "prompt": Value("string"),
        "first_frame": Image(),
        "final_frame": Image(),
        "ground_truth_video": Value("binary"),
    }
)

_README_TEMPLATE = """\
---
language:
- en
license: apache-2.0
task_categories:
- visual-question-answering
- video-classification
tags:
- video
- reasoning
- benchmark
- i2v
pretty_name: VBVR-Bench
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: in_domain
    path: data/in_domain-*
  - split: out_of_domain
    path: data/out_of_domain-*
---

# VBVR-Bench

Re-hosted copy of [Video-Reason/VBVR-Bench-Data](https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data),
converted to standard HuggingFace parquet format.

## Splits
- **`in_domain`**: 50 tasks x 5 samples = 250 entries (tasks overlap with the VBVR training set).
- **`out_of_domain`**: 50 tasks x 5 samples = 250 entries (held-out reasoning tasks).

## Schema

| field | type | notes |
|---|---|---|
| `task_name` | string | e.g. `G-13_grid_number_sequence_data-generator` |
| `video_idx` | string | zero-padded sample id (`00000`..`00004`) |
| `domain` | string | duplicates split name; convenient for filtering |
| `prompt` | string | task description fed to the I2V model |
| `first_frame` | Image (PNG) | I2V condition frame |
| `final_frame` | Image (PNG) | expected final frame |
| `ground_truth_video` | binary (MP4) | reference video — decode with decord / PyAV |

## Quick load

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="in_domain")
sample = ds[0]
sample["first_frame"]          # PIL.Image
sample["prompt"]               # str
sample["ground_truth_video"]   # raw MP4 bytes

# Decode the video with decord
import decord, io
vr = decord.VideoReader(io.BytesIO(sample["ground_truth_video"]))
```

## Links
- Upstream dataset: [Video-Reason/VBVR-Bench-Data](https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data)
- Evaluation kit: [Video-Reason/VBVR-EvalKit](https://github.com/Video-Reason/VBVR-EvalKit)
- Project page: [video-reason.com](https://video-reason.com/)
"""


def _iter_samples(domain_dir: Path, domain_label: str) -> Iterator[dict]:
    if not domain_dir.is_dir():
        raise FileNotFoundError(f"missing domain dir: {domain_dir}")
    for task_dir in sorted(domain_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for sample_dir in sorted(task_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            first = sample_dir / "first_frame.png"
            final = sample_dir / "final_frame.png"
            video = sample_dir / "ground_truth.mp4"
            prompt_path = sample_dir / "prompt.txt"
            if not all(p.exists() for p in (first, final, video, prompt_path)):
                logger.warning("skip incomplete sample: {}", sample_dir)
                continue
            yield {
                "task_name": task_dir.name,
                "video_idx": sample_dir.name,
                "domain": domain_label,
                "prompt": prompt_path.read_text(encoding="utf-8").strip(),
                "first_frame": str(first),
                "final_frame": str(final),
                "ground_truth_video": video.read_bytes(),
            }


def _build_split(gt_base: Path, dir_name: str, split_name: str) -> Dataset:
    samples = list(_iter_samples(gt_base / dir_name, split_name))
    logger.info("split '{}': {} samples", split_name, len(samples))
    return Dataset.from_list(samples, features=_FEATURES)


def build_dataset(gt_base: Path) -> DatasetDict:
    return DatasetDict(
        {split_name: _build_split(gt_base, dir_name, split_name) for dir_name, split_name in _DOMAIN_DIRS.items()}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt_base", type=Path, required=True, help="Dir with In-Domain_50/ and Out-of-Domain_50/")
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="Optional local save path (Arrow format via save_to_disk)"
    )
    parser.add_argument(
        "--push_to_hub", type=str, default=None, help="HF dataset repo id to push to, e.g. pufanyi/VBVR-Bench"
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")
    parser.add_argument(
        "--token", type=str, default=None, help="HF token; if omitted uses HF_TOKEN env or cached login"
    )
    parser.add_argument("--max_shard_size", type=str, default="500MB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds = build_dataset(args.gt_base)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(args.output_dir))
        logger.info("saved to disk: {}", args.output_dir)

    if args.push_to_hub is not None:
        logger.info("pushing to hub: {}", args.push_to_hub)
        ds.push_to_hub(
            args.push_to_hub,
            private=args.private,
            token=args.token,
            max_shard_size=args.max_shard_size,
        )
        HfApi().upload_file(
            path_or_fileobj=_README_TEMPLATE.format(repo_id=args.push_to_hub).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.push_to_hub,
            repo_type="dataset",
            token=args.token,
            commit_message="Add dataset card",
        )
        logger.info("uploaded README. done.")


if __name__ == "__main__":
    main()
