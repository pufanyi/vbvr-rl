"""Upload the precomputed VBVR-Bench WebDataset to HuggingFace.

Expects ``--tar_dir`` to contain ``shard-*.tar`` shards produced by
``scripts/eval/pre_compute_vbvr_bench.fish``. Uploads them under ``tars/`` in
the target repo and writes a README card describing the format.

Deps:
    uv add huggingface_hub

Usage:
    .venv/bin/python scripts/data/vbvr_to_hf.py \\
        --tar_dir data/vbvr/VBVR-Bench-wds \\
        --repo_id pufanyi/VBVR-Bench
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger

_README_TEMPLATE = """\
---
license: apache-2.0
task_categories:
- visual-question-answering
- video-classification
tags:
- video
- reasoning
- benchmark
- i2v
- webdataset
- precomputed
pretty_name: VBVR-Bench
size_categories:
- n<1K
---

# VBVR-Bench (precomputed WebDataset)

Eval set for visual reasoning in video generation, re-packaged as a
[WebDataset](https://github.com/webdataset/webdataset) of tar shards with
T5 prompt embeddings and VAE-encoded first-frame condition tensors baked in
for the Wan2.2-I2V-A14B pipeline. Raw first/final frames and the reference
video are shipped alongside so downstream scorers (VLM judges or rule-based
evaluators) can read them without going back to the source GT tree.

## Layout

```
tars/
├── shard-000000.tar
├── shard-000001.tar
└── ...
```

Each tar shard contains, per sample (key = ``{{split}}_{{task_name}}_{{video_idx}}``):

| suffix | content |
|---|---|
| `.safetensors` | bf16 tensors: `prompt_embeds`, `condition` |
| `.json` | metadata: `task_name`, `video_idx`, `split`, `domain`, `prompt`, `height`, `width`, `num_frames` |
| `.first.png` | first frame (I2V condition image) |
| `.final.png` | expected final frame |
| `.gt.mp4` | ground-truth reference video |

## Splits
- `In_Domain` (250 samples): 50 tasks overlapping with the VBVR training set.
- `Out_of_Domain` (250 samples): 50 held-out reasoning tasks.
- Filter on the `domain` field in the per-sample JSON — the tar shards are domain-mixed.

## Quick load with webdataset

```python
import webdataset as wds

url = "hf://datasets/{repo_id}/tars/shard-{{000000..000004}}.tar"
ds = wds.WebDataset(url).decode()
for sample in ds:
    meta = sample["json"]
    prompt_embeds = sample["safetensors"]["prompt_embeds"]
    condition = sample["safetensors"]["condition"]
    first_frame = sample["first.png"]        # PIL.Image via decode()
    gt_video_bytes = sample["gt.mp4"]        # raw bytes — decode with decord / PyAV
```

## Generation with datasets

```python
from datasets import load_dataset

ds = load_dataset("webdataset", data_files="hf://datasets/{repo_id}/tars/*.tar", split="train")
```

## Source
- Upstream GT: [Video-Reason/VBVR-Bench-Data](https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data)
- Evaluation kit: [Video-Reason/VBVR-EvalKit](https://github.com/Video-Reason/VBVR-EvalKit)
- Model used for encoding: `Wan2.2-I2V-A14B-Diffusers`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tar_dir", type=Path, required=True, help="Directory with shard-*.tar files")
    parser.add_argument("--repo_id", type=str, default="pufanyi/VBVR-Bench-wan2.2-latent")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", type=str, default=None, help="HF token; else uses HF_TOKEN env / cached login")
    parser.add_argument("--commit_message", type=str, default="Upload VBVR-Bench webdataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    shards = sorted(args.tar_dir.glob("shard-*.tar"))
    if not shards:
        raise FileNotFoundError(f"no shard-*.tar under {args.tar_dir}")
    total_mb = sum(p.stat().st_size for p in shards) / 1e6
    logger.info(
        "uploading {} shards ({:.1f} MB) from {} -> {}",
        len(shards),
        total_mb,
        args.tar_dir,
        args.repo_id,
    )

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    api.upload_folder(
        folder_path=str(args.tar_dir),
        path_in_repo="tars",
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["shard-*.tar"],
        commit_message=args.commit_message,
    )

    api.upload_file(
        path_or_fileobj=_README_TEMPLATE.format(repo_id=args.repo_id).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    logger.info("done -> https://huggingface.co/datasets/{}", args.repo_id)


if __name__ == "__main__":
    main()
