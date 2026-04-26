"""Wan2.2 I2V batch evaluation — generate videos for a dataset.

Multi-GPU data-parallel: each rank loads its own pipeline and processes
a disjoint slice of the dataset.

Usage:
    # Single GPU
    .venv/bin/python -m src.cli.eval_i2v --eval_json data/eval.json --output_dir eval_out/

    # Multi-GPU
    .venv/bin/torchrun --nproc_per_node=8 -m src.cli.eval_i2v \
        --eval_json data/eval.json --output_dir eval_out/

JSON format (compatible with training dataset):
[
    {"image": "path/to/image.jpg", "prompt": "a cat playing"},
    {"image": "img2.jpg", "prompt": "sunset over the ocean", "video": "ignored.mp4"},
    ...
]

If "image" is absent but "video" is present, the first frame of the video is used.
"""

import argparse
import gc
import json
import os
from datetime import timedelta
from pathlib import Path

import decord
import numpy as np
import torch
import torch.distributed as dist
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 I2V batch evaluation")
    parser.add_argument("--eval_json", type=str, required=True, help="JSON file with eval samples")
    parser.add_argument(
        "--model_path",
        type=str,
        default="storage/models/Wan2.2-I2V-A14B-Diffusers",
        help="Path to the model directory",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a DCP training checkpoint directory",
    )
    parser.add_argument("--use_ema", action="store_true", help="Load EMA shadow weights from DCP checkpoint")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative prompt",
    )
    parser.add_argument(
        "--max_area",
        type=int,
        default=480 * 832,
        help="Max pixel area (default 480*832 for 480P)",
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--parallel_load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the model concurrently on all ranks. Use --no-parallel_load to reduce host RAM peak.",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        choices=["nccl"],
        help="Distributed backend for rank coordination.",
    )
    parser.add_argument(
        "--dist_timeout_minutes",
        type=int,
        default=180,
        help="Process-group timeout in minutes. Slow serialized model loading can exceed PyTorch's default.",
    )
    return parser.parse_args()


def _resolve_path(path: str, base_dir: Path) -> str:
    p = Path(path)
    return str(p) if p.is_absolute() else str(base_dir / p)


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _first_frame_from_video(video_path: str) -> Image.Image:
    vr = decord.VideoReader(video_path)
    frame = vr[0].numpy()
    return Image.fromarray(frame)


def _load_pipeline(args, device: torch.device, rank: int) -> WanImageToVideoPipeline:
    if rank == 0:
        print(f"Loading model from {args.model_path} ...", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.to(device)
    gc.collect()

    if args.checkpoint:
        from src.trainer.checkpoint import load_dcp_into_pipeline

        if rank == 0:
            print(f"Loading DCP checkpoint from {args.checkpoint} (ema={args.use_ema}) ...", flush=True)
        load_dcp_into_pipeline(pipe, args.checkpoint, use_ema=args.use_ema)
        gc.collect()

    return pipe


def _barrier(device: torch.device) -> None:
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def _load_pipeline_rank_serialized(args, device: torch.device, rank: int, world_size: int) -> WanImageToVideoPipeline:
    if world_size == 1 or args.parallel_load:
        return _load_pipeline(args, device, rank)

    pipe = None
    if rank == 0:
        print("Loading model rank-by-rank to avoid host RAM spikes.", flush=True)
    for load_rank in range(world_size):
        if rank == load_rank:
            print(f"[rank {rank}] Loading pipeline on {device} ...", flush=True)
            pipe = _load_pipeline(args, device, rank)
            print(f"[rank {rank}] Pipeline ready on {device}", flush=True)
        _barrier(device)
    assert pipe is not None
    return pipe


def main():
    args = parse_args()

    # ---- Distributed setup (works for both single and multi-GPU) ----
    if "RANK" in os.environ:
        dist.init_process_group(args.dist_backend, timeout=timedelta(minutes=args.dist_timeout_minutes))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ---- Load eval data ----
    eval_json = Path(args.eval_json)
    base_dir = eval_json.parent
    data = json.loads(eval_json.read_text())

    # Partition across ranks (round-robin for balanced load)
    my_indices = list(range(rank, len(data), world_size))
    if rank == 0:
        print(f"Eval: {len(data)} samples, {world_size} GPUs, {len(my_indices)} samples on this rank")

    # ---- Load pipeline ----
    pipe = _load_pipeline_rank_serialized(args, device, rank, world_size)

    # ---- Output dir ----
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        _barrier(device)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolution helper ----
    mod_value = pipe.vae_scale_factor_spatial * (
        pipe.transformer.config.patch_size[1]
        if pipe.transformer is not None
        else pipe.transformer_2.config.patch_size[1]
    )

    # ---- Generate ----
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for count, idx in enumerate(my_indices):
        item = data[idx]

        # Determine output filename
        name = item.get("name") or item.get("id") or str(idx)
        out_path = output_dir / f"{name}.mp4"
        if out_path.exists():
            print(f"[rank {rank}] Skipping {out_path} (exists)")
            continue

        # Load reference image
        raw_image = item.get("image")
        if isinstance(raw_image, list):
            raw_image = raw_image[0] if raw_image else None
        if raw_image:
            image = _load_image(_resolve_path(raw_image, base_dir))
        elif "video" in item:
            image = _first_frame_from_video(_resolve_path(item["video"], base_dir))
        else:
            print(f"[rank {rank}] Skipping index {idx}: no image or video")
            continue

        prompt = item["prompt"]

        # Compute resolution
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(args.max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(args.max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        # Generate
        frames = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).frames[0]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(out_path), fps=args.fps)
        print(f"[rank {rank}] [{count + 1}/{len(my_indices)}] Saved {out_path}")

    if world_size > 1:
        _barrier(device)
        dist.destroy_process_group()

    if rank == 0:
        print(f"Done. Generated videos in {output_dir}")


if __name__ == "__main__":
    main()
