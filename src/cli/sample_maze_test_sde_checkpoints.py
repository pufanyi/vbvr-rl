"""Sample one raw maze test item with SDE across multiple DCP checkpoints.

This utility is for quick visual comparison of dense DanceGRPO checkpoints. It
loads the 5B TI2V model once, encodes one raw eval-json sample, then reloads
each DCP checkpoint into the transformer and saves multiple SDE rollouts.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.cli.sample_dancegrpo_sde import (
    _export_uint8_video,
    _load_checkpoint_into_model,
    _load_config,
    _pairwise_video_metrics,
    _save_contact_sheet,
    _torch_dtype,
)
from src.models.wan_i2v import WanI2VForTraining
from src.precompute.maze_webdataset import (
    COS_CHAIN_MODE_LINE_TO_MOVING_BALL,
    GenConfig,
    _build_maze_specs,
    _generate_shard_samples,
    _sample_to_json_blob,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train_dancegrpo_maze_5b_line_to_ball_rl_mse_2.yaml")
    p.add_argument("--eval_json", default=None)
    p.add_argument(
        "--generated_maze_gid",
        type=int,
        default=None,
        help="Generate a same-style holdout maze with dataset seed + gid. Use e.g. 100000 to avoid train split.",
    )
    p.add_argument("--maze_seed", type=int, default=4242)
    p.add_argument("--maze_cell_h", type=int, default=16)
    p.add_argument("--maze_cell_w", type=int, default=16)
    p.add_argument("--maze_cell_px", type=int, default=12)
    p.add_argument("--maze_difficulty_names", default="easy,mid,hard,xhard")
    p.add_argument(
        "--checkpoint_root",
        default="storage/checkpoints/dancegrpo_maze_5b_line_to_ball_rl_mse_shared_prompt_2",
    )
    p.add_argument("--output_dir", default="storage/outputs/maze_test_sde_mse_shared_prompt_2")
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--sample_batch_size", type=int, default=1)
    p.add_argument("--num_sampling_steps", type=int, default=None)
    p.add_argument("--sde_noise_scale", type=float, default=None)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--num_frames", type=int, default=161)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--transformer_dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--checkpoint_prefer", choices=["auto", "raw", "ema"], default="raw")
    p.add_argument("--checkpoint", action="append", default=None, help="Specific checkpoint path; repeatable")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def _resolve_path(path: str, base_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base_dir / p


def _load_eval_sample(eval_json: str, sample_index: int) -> tuple[dict[str, Any], Path]:
    eval_path = Path(eval_json)
    data = json.loads(eval_path.read_text())
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected eval json to contain a list or dict: {eval_json}")
    if sample_index < 0 or sample_index >= len(data):
        raise IndexError(f"sample_index={sample_index} out of range for {eval_json} ({len(data)} samples)")
    sample = data[sample_index]
    if not isinstance(sample, dict):
        raise ValueError(f"Sample {sample_index} in {eval_json} is not an object")
    if "prompt" not in sample:
        raise ValueError(f"Sample {sample_index} in {eval_json} has no prompt")
    if not (sample.get("first_frame") or sample.get("image")):
        raise ValueError(f"Sample {sample_index} in {eval_json} has no first_frame or image")
    return sample, eval_path.parent


def _load_generated_maze_sample(
    args: argparse.Namespace,
    model_path: str,
) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray]]:
    if args.generated_maze_gid is None:
        raise ValueError("generated_maze_gid is required")
    if args.generated_maze_gid < 0:
        raise ValueError(f"generated_maze_gid must be non-negative, got {args.generated_maze_gid}")

    gen_cfg = GenConfig(
        output_dir=str(Path(args.output_dir) / "_unused_generation"),
        num_samples=max(args.generated_maze_gid + 1, 1),
        model_path=model_path,
        cell_h=args.maze_cell_h,
        cell_w=args.maze_cell_w,
        cell_px=args.maze_cell_px,
        num_frames=args.num_frames,
        difficulty_names=args.maze_difficulty_names,
        render_mode="moving_ball",
        cos_chain_mode=COS_CHAIN_MODE_LINE_TO_MOVING_BALL,
        seed=args.maze_seed,
        num_preview_videos=0,
        preview_fps=args.fps,
    )
    specs, (height, width), _geometry_map = _build_maze_specs(gen_cfg)
    if height != args.height or width != args.width:
        raise ValueError(
            "Generated maze geometry does not match requested resolution: "
            f"geometry gives {width}x{height}, requested {args.width}x{args.height}"
        )
    generated = _generate_shard_samples(gen_cfg, specs, [args.generated_maze_gid], "test")
    videos, samples, gid = generated[0]
    final_sample = samples[-1]
    metadata = {
        "global_index": gid,
        "split": "test_generated_holdout",
        "prompt": final_sample.prompt,
        "num_latents": len(videos),
        "cos_chain_mode": COS_CHAIN_MODE_LINE_TO_MOVING_BALL,
        "maze": _sample_to_json_blob(final_sample, fps=args.fps),
    }
    if len(samples) > 1:
        metadata["maze_chain"] = [_sample_to_json_blob(sample, fps=args.fps) for sample in samples]
    return metadata, videos[-1][0], videos


def _load_image_tensor(path: Path, height: int, width: int, device: torch.device) -> torch.Tensor:
    image = Image.open(str(path)).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)


def _image_array_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image array, got shape {image.shape}")
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)


def _discover_checkpoints(root: str, explicit: list[str] | None) -> list[Path]:
    if explicit:
        checkpoints = [Path(p) for p in explicit]
    else:
        root_path = Path(root)
        checkpoints = []
        for metadata in root_path.glob("checkpoint-*/high/.metadata"):
            checkpoints.append(metadata.parent.parent)
        for metadata in root_path.glob("checkpoint-*/.metadata"):
            checkpoints.append(metadata.parent)

    unique = {str(p): p for p in checkpoints}

    def step_key(path: Path) -> tuple[int, str]:
        name = path.name
        if name.startswith("checkpoint-"):
            tail = name[len("checkpoint-") :]
            if tail.isdigit():
                return int(tail), str(path)
        return 10**18, str(path)

    return sorted(unique.values(), key=step_key)


def _decode_batch_to_uint8(model: WanI2VForTraining, latents: torch.Tensor) -> np.ndarray:
    decoded = model.decode_latents(latents)
    decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return decoded.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()


def _checkpoint_output_name(checkpoint: Path) -> str:
    return checkpoint.name


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _discover_checkpoints(args.checkpoint_root, args.checkpoint)
    if not checkpoints:
        raise FileNotFoundError(f"No DCP checkpoints found under {args.checkpoint_root}")

    generated_videos: list[np.ndarray] = []
    first_frame_path: Path | None = None
    solution_path: Path | None = None
    first_frame_array: np.ndarray | None = None
    if args.generated_maze_gid is not None:
        sample, first_frame_array, generated_videos = _load_generated_maze_sample(args, cfg.model_path)
    else:
        if args.eval_json is None:
            raise ValueError("Either --generated_maze_gid or --eval_json is required")
        sample, sample_base = _load_eval_sample(args.eval_json, args.sample_index)
        first_frame_path = _resolve_path(str(sample.get("first_frame") or sample.get("image")), sample_base)
        solution_path = _resolve_path(str(sample["image"]), sample_base) if sample.get("image") else None

    group_size = int(args.group_size)
    sample_batch_size = max(1, int(args.sample_batch_size))
    num_sampling_steps = args.num_sampling_steps or cfg.grpo_num_sampling_steps
    sde_noise_scale = args.sde_noise_scale if args.sde_noise_scale is not None else cfg.grpo_sde_noise_scale
    sde_formula = getattr(cfg, "grpo_sde_formula", "dancegrpo")
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.grpo_cfg_scale
    seed = args.seed if args.seed is not None else cfg.seed

    if args.generated_maze_gid is not None:
        print(
            f"[sample] generated_holdout gid={args.generated_maze_gid} seed={args.maze_seed + args.generated_maze_gid}",
            flush=True,
        )
    else:
        print(f"[sample] eval_json={args.eval_json} index={args.sample_index}", flush=True)
    print(f"[sample] prompt={sample['prompt']}", flush=True)
    if first_frame_path is not None:
        print(f"[sample] first_frame={first_frame_path}", flush=True)
    print(f"[checkpoints] {len(checkpoints)} total: {[p.name for p in checkpoints]}", flush=True)
    print(
        f"[sampling] groups={group_size} batch={sample_batch_size} T={num_sampling_steps} "
        f"formula={sde_formula} eta={sde_noise_scale} cfg={cfg_scale}",
        flush=True,
    )

    model = WanI2VForTraining(
        cfg.model_path,
        train_experts="both",
        train_text_encoder=False,
        gradient_checkpointing=False,
        load_vae=True,
        load_text_encoder=True,
        transformer_dtype=_torch_dtype(args.transformer_dtype),
    )
    model.transformer.to(device)
    model.transformer.eval()
    if model.vae is None or model.text_encoder is None:
        raise RuntimeError("Expected VAE and text encoder to be loaded")
    model.vae.to(device).eval()
    model.text_encoder.to(device).eval()

    if first_frame_array is not None:
        image = _image_array_to_tensor(first_frame_array, device)
    elif first_frame_path is not None:
        image = _load_image_tensor(first_frame_path, args.height, args.width, device)
    else:
        raise RuntimeError("No first frame source was prepared")
    with torch.no_grad():
        prompt_embeds = model.encode_text([str(sample["prompt"])], device=device)
        condition = model.prepare_condition(image, args.num_frames, args.height, args.width)
    del image
    model.text_encoder = None
    model.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

    ref_dir = output_dir / "sample_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    if first_frame_array is not None:
        Image.fromarray(first_frame_array).save(ref_dir / "first_frame.png")
    elif first_frame_path is not None:
        shutil.copy2(str(first_frame_path), str(ref_dir / "first_frame.png"))
    if solution_path is not None and solution_path.exists():
        shutil.copy2(str(solution_path), str(ref_dir / "solution.png"))
    for idx, ref_video in enumerate(generated_videos):
        suffix = "moving_ball" if idx == len(generated_videos) - 1 else "line_waypoint"
        _export_uint8_video(ref_video, ref_dir / f"reference_{idx}_{suffix}.mp4", args.fps)
    (ref_dir / "sample.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")

    run_manifest: dict[str, Any] = {
        "config": args.config,
        "eval_json": args.eval_json,
        "generated_maze_gid": args.generated_maze_gid,
        "maze_seed": args.maze_seed,
        "sample_index": args.sample_index,
        "sample": sample,
        "checkpoint_root": args.checkpoint_root,
        "output_dir": str(output_dir),
        "sampling": {
            "group_size": group_size,
            "sample_batch_size": sample_batch_size,
            "num_sampling_steps": num_sampling_steps,
            "sde_formula": sde_formula,
            "sde_noise_scale": sde_noise_scale,
            "cfg_scale": cfg_scale,
            "share_group_init_noise": cfg.dancegrpo_share_group_init_noise,
            "seed": seed,
            "fps": args.fps,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
        },
        "checkpoints": [],
    }

    for ckpt_idx, checkpoint in enumerate(checkpoints, start=1):
        ckpt_out = output_dir / _checkpoint_output_name(checkpoint)
        manifest_path = ckpt_out / "manifest.json"
        videos_expected = [ckpt_out / f"group_{g:02d}.mp4" for g in range(group_size)]
        if manifest_path.exists() and all(path.exists() for path in videos_expected) and not args.force:
            print(f"[checkpoint {ckpt_idx}/{len(checkpoints)}] skipping existing {checkpoint}", flush=True)
            run_manifest["checkpoints"].append(json.loads(manifest_path.read_text()))
            continue

        ckpt_out.mkdir(parents=True, exist_ok=True)
        print(f"[checkpoint {ckpt_idx}/{len(checkpoints)}] loading {checkpoint}", flush=True)
        ckpt_info = _load_checkpoint_into_model(model, str(checkpoint), prefer=args.checkpoint_prefer)
        print(f"[checkpoint {ckpt_idx}/{len(checkpoints)}] {ckpt_info}", flush=True)

        init_generator = torch.Generator(device=device).manual_seed(seed + 17)
        shared_initial_latent = None
        if cfg.dancegrpo_share_group_init_noise:
            latent_shape = model.latent_shape_from_condition(condition)
            shared_initial_latent = torch.randn(
                latent_shape,
                device=device,
                dtype=torch.bfloat16,
                generator=init_generator,
            )

        videos: list[np.ndarray] = []
        video_paths: list[str] = []
        started = time.time()
        for group_start in range(0, group_size, sample_batch_size):
            cur_s = min(sample_batch_size, group_size - group_start)
            out_paths = [ckpt_out / f"group_{g:02d}.mp4" for g in range(group_start, group_start + cur_s)]
            cond_s = condition.repeat_interleave(cur_s, dim=0)
            pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0)
            initial_latent = (
                shared_initial_latent.repeat_interleave(cur_s, dim=0) if shared_initial_latent is not None else None
            )
            rollout_generator = torch.Generator(device=device).manual_seed(seed + 1009 * (group_start + 1))
            print(
                f"[checkpoint {ckpt_idx}/{len(checkpoints)}] groups {group_start}..{group_start + cur_s - 1}",
                flush=True,
            )
            with torch.no_grad():
                traj = model.sde_generate(
                    condition=cond_s,
                    prompt_embeds=pe_s,
                    num_sampling_steps=num_sampling_steps,
                    sde_noise_scale=sde_noise_scale,
                    sigma_min=cfg.grpo_sde_sigma_min,
                    sigma_max=cfg.grpo_sde_sigma_max,
                    cfg_scale=cfg_scale,
                    generator=rollout_generator,
                    initial_latent=initial_latent,
                    sde_formula=sde_formula,
                )
                decoded_uint8 = _decode_batch_to_uint8(model, traj["latents"][-1])

            for local_idx, out_path in enumerate(out_paths):
                video = decoded_uint8[local_idx]
                _export_uint8_video(video, out_path, args.fps)
                videos.append(video)
                video_paths.append(str(out_path))

            del traj, decoded_uint8, cond_s, pe_s, initial_latent
            torch.cuda.empty_cache()

        metrics = _pairwise_video_metrics(videos) if videos else {}
        if videos:
            _save_contact_sheet(videos, ckpt_out / "contact_sheet.jpg")

        checkpoint_manifest = {
            "checkpoint": str(checkpoint),
            "checkpoint_info": ckpt_info,
            "output_dir": str(ckpt_out),
            "videos": video_paths,
            "metrics": metrics,
            "elapsed_seconds": time.time() - started,
        }
        manifest_path.write_text(json.dumps(checkpoint_manifest, indent=2), encoding="utf-8")
        run_manifest["checkpoints"].append(checkpoint_manifest)
        del videos
        gc.collect()
        torch.cuda.empty_cache()

    (output_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote results to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
