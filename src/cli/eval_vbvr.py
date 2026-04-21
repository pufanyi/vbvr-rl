"""CLI for VBVR-Bench VLM-judged evaluation.

Usage:
    uv run python -m src.cli.eval_vbvr \\
        --model_output storage/eval_out/vbvr/sft_maze_checkpoint-2000 \\
        --gt_base /mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench \\
        --output_dir storage/eval_out/vbvr_vlm

The generated videos must already exist at
<model_output>/{Open_60,Hidden_40}/<task>/<idx>.mp4 — produce them with
`src.cli.eval_i2v` (see scripts/eval/eval_vbvr.fish).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from src.eval.vbvr.judges import VLMJudge
from src.eval.vbvr.runner import run_eval

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VBVR-Bench VLM-judged evaluation")
    parser.add_argument(
        "--model_output",
        type=Path,
        required=True,
        help="Dir containing Open_60/ and/or Hidden_40/ with generated .mp4 videos",
    )
    parser.add_argument(
        "--gt_base",
        type=Path,
        default=Path("data/vbvr/VBVR-Bench"),
        help="Base dir with In-Domain_50/ and Out-of-Domain_50/",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("storage/eval_out/vbvr_vlm"),
        help="Directory to write <model_name>/eval_results.json + summary.json",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="google/gemma-4-26B-A4B-it",
        help="HF model id for the VLM judge (image-text-to-text)",
    )
    parser.add_argument("--num_frames", type=int, default=6, help="Frames uniformly sampled from each generated video")
    parser.add_argument("--include_gt_first_frame", action="store_true", help="Also show the starting frame to the VLM")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(_DTYPE_MAP))
    parser.add_argument(
        "--device_map", type=str, default=None, help="Pass to HF from_pretrained (e.g. 'auto' for model parallel)"
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Only score these task names")
    parser.add_argument("--limit", type=int, default=None, help="Score at most N samples (for quick smoke tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Model output:    {}", args.model_output)
    logger.info("GT base:         {}", args.gt_base)
    logger.info("Judge model:     {}", args.judge_model)
    logger.info("Num frames:      {}", args.num_frames)

    judge = VLMJudge(
        model_id=args.judge_model,
        num_frames=args.num_frames,
        include_gt_first_frame=args.include_gt_first_frame,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=_DTYPE_MAP[args.dtype],
        device_map=args.device_map,
    )

    run_eval(
        model_output=args.model_output,
        gt_base=args.gt_base,
        judge=judge,
        output_dir=args.output_dir,
        tasks=args.tasks,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
