"""CLI for VBVR-Bench VLM-judged evaluation.

Single GPU:
    .venv/bin/python -m src.cli.eval_vbvr \\
        --model_output storage/eval_out/vbvr/sft_maze_checkpoint-2000

Multi-GPU data-parallel (one VLM copy per rank, per-rank JSONL shards):
    .venv/bin/torchrun --nproc_per_node=8 -m src.cli.eval_vbvr \\
        --model_output storage/eval_out/vbvr/sft_maze_checkpoint-2000

Videos must already exist at <model_output>/{Open_60,Hidden_40}/<task>/<idx>.mp4
— generate them with ``src.cli.eval_i2v`` (see ``scripts/eval/vbvr_generate_score.fish``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
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
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(_DTYPE_MAP))
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="HF model parallel (e.g. 'auto'). Only for single-process runs; incompatible with torchrun DP.",
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Only score these task names")
    parser.add_argument("--limit", type=int, default=None, help="Score at most N samples (for quick smoke tests)")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Rank 0 backs up existing scores.rank*.jsonl shards; score everything again",
    )
    parser.add_argument(
        "--retry_errors",
        action="store_true",
        help="Re-score cached samples whose prior attempt errored",
    )
    return parser.parse_args()


def _init_distributed() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank). Init NCCL group if torchrun-launched."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = _init_distributed()

    if args.device_map is not None and world_size > 1:
        raise ValueError("--device_map is incompatible with torchrun DP; pick one")

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if rank == 0:
        logger.info("Model output:    {}", args.model_output)
        logger.info("GT base:         {}", args.gt_base)
        logger.info("Judge model:     {}", args.judge_model)
        logger.info("Num frames:      {}", args.num_frames)
        logger.info("World size:      {}", world_size)

    judge = VLMJudge(
        model_id=args.judge_model,
        num_frames=args.num_frames,
        include_gt_first_frame=args.include_gt_first_frame,
        max_new_tokens=args.max_new_tokens,
        device=device,
        torch_dtype=_DTYPE_MAP[args.dtype],
        device_map=args.device_map,
    )

    if world_size > 1:
        backend = dist.get_backend()
        barrier_fn = (lambda: dist.barrier(device_ids=[local_rank])) if backend == "nccl" else dist.barrier
    else:
        barrier_fn = None
    run_eval(
        model_output=args.model_output,
        gt_base=args.gt_base,
        judge=judge,
        output_dir=args.output_dir,
        tasks=args.tasks,
        limit=args.limit,
        fresh=args.fresh,
        retry_errors=args.retry_errors,
        rank=rank,
        world_size=world_size,
        barrier_fn=barrier_fn,
    )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
