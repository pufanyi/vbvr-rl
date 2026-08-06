"""Render all clean-endpoint previews for one formal VBVR I2V sample.

The CPS path calls the same training-time Flow-CPS engine as
``src.cli.eval_i2v_cps``.  Euler and UniPC call the same Diffusers pipeline as
``src.cli.eval_i2v_euler`` / ``src.cli.eval_i2v`` and intercept the scheduler
immediately before each real solver step.  Thus every preview has the same
meaning: the post-CFG flow clean endpoint ``x0 = x_t - sigma * v``.  For the
last displayed cell we decode the solver's actual final latent at sigma zero.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import decord
import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from src.cli import eval_i2v
from src.cli.eval_i2v_cps import _inference_config, _prepare_input
from src.cli.eval_i2v_euler import install_flowmatch_euler_scheduler
from src.inference.engine import InferenceEngine, build_model
from src.inference.inputs import PreparedInput
from src.inference.outputs import save_step_contact_sheet, save_step_grid_video, write_outputs


def _default_negative_prompt() -> str:
    return eval_i2v.parse_args(["--eval_json", "unused", "--output_dir", "unused"]).negative_prompt


def _noise_level(value: str) -> float:
    level = float(value)
    if not 0.0 <= level <= 1.0:
        raise argparse.ArgumentTypeError(f"CPS noise level must be in [0, 1], got {value}")
    return level


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--model_path", required=True, help="Converted Diffusers model directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sampler", required=True, choices=("cps", "euler", "unipc"))
    parser.add_argument("--noise_level", type=_noise_level, default=None, help="Required only for CPS")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative_prompt", default=_default_negative_prompt())
    parser.add_argument(
        "--formal_final_video",
        default=None,
        help="Bind step T/final byte-for-byte to the video actually used by the quantitative evaluation",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid_cols", type=int, default=6)
    parser.add_argument("--grid_thumb_width", type=int, default=160)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.sampler == "cps" and args.noise_level is None:
        parser.error("--noise_level is required when --sampler=cps")
    if args.sampler != "cps" and args.noise_level is not None:
        parser.error("--noise_level is only valid when --sampler=cps")
    if args.sample_index < 0:
        parser.error("--sample_index must be non-negative")
    if args.num_inference_steps < 1:
        parser.error("--num_inference_steps must be positive")
    return args


def _load_sample(args: argparse.Namespace) -> tuple[dict[str, Any], Image.Image, str, str]:
    eval_args = SimpleNamespace(eval_json=args.eval_json, limit=None)
    _, base_dir, data = eval_i2v._load_eval_data(eval_args)
    if args.sample_index >= len(data):
        raise IndexError(f"sample_index={args.sample_index} is out of range for {len(data)} eval samples")
    item = data[args.sample_index]
    name = eval_i2v._sample_name(item, args.sample_index)

    raw_image = item.get("image")
    if isinstance(raw_image, list):
        raw_image = raw_image[0] if raw_image else None
    if raw_image:
        image = eval_i2v._load_image(eval_i2v._resolve_path(raw_image, base_dir))
    elif item.get("video"):
        image = eval_i2v._first_frame_from_video(eval_i2v._resolve_path(item["video"], base_dir))
    else:
        raise ValueError(f"Eval item {args.sample_index} has no image or video")

    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Eval item {args.sample_index} has no prompt")
    image = image.resize((args.width, args.height), Image.Resampling.LANCZOS)
    return item, image, prompt, name


def _render_cps(
    args: argparse.Namespace,
    *,
    item: dict[str, Any],
    image: Image.Image,
    prompt: str,
    name: str,
    device: torch.device,
) -> dict[str, Any]:
    cps_args = SimpleNamespace(
        model_path=args.model_path,
        checkpoint=None,
        use_ema=False,
        output_dir=args.output_dir,
        noise_level=args.noise_level,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
        seed=args.seed,
    )
    sample_seed = args.seed + args.sample_index
    cfg = _inference_config(cps_args, device, seed=sample_seed).model_copy(
        update={
            "image": item.get("image") or item.get("video") or "eval-item",
            "prompt": prompt,
            "save_steps": True,
            "save_reference": False,
            "grid_cols": args.grid_cols,
            "grid_thumb_width": args.grid_thumb_width,
            "force": args.force,
        }
    )
    model = build_model(cfg, need_text_encoder=True)
    prepared_base = _prepare_input(model, image, prompt, cps_args, device)
    prepared = PreparedInput(
        condition=prepared_base.condition,
        prompt_embeds=prepared_base.prompt_embeds,
        source=f"vbvr-eval-json:{name}",
        metadata={"eval_json": args.eval_json, "sample_index": args.sample_index, "name": name},
        summary={"name": name, "sample_index": args.sample_index, "prompt": prompt},
    )
    result = InferenceEngine(model, cfg).sample(prepared)
    if len(result.pred_x0) != args.num_inference_steps:
        raise RuntimeError(f"Captured {len(result.pred_x0)} CPS previews, expected {args.num_inference_steps}")
    manifest = write_outputs(model, cfg, prepared, result, Path(args.output_dir), started=time.time())
    formal_binding = _bind_formal_final(args, Path(args.output_dir))
    manifest.update(
        {
            "sampler": "flow_cps",
            "sampler_implementation": "src.cli.eval_i2v_cps / WanI2VForTraining.sde_generate",
            "eval_json": str(Path(args.eval_json).resolve()),
            "sample_index": args.sample_index,
            "sample_name": name,
            "preview_contract": "post-CFG x0=x_t-sigma*v; final cell is actual solver latent at sigma=0",
            "formal_final_binding": formal_binding,
        }
    )
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class _OdeStepRecorder:
    """Intercept a Diffusers scheduler without changing its numerical step."""

    def __init__(self, pipe: Any):
        self.pipe = pipe
        self.pred_x0: list[torch.Tensor] = []
        self.sigmas: list[float] = []
        self.timesteps: list[float] = []
        self.condition: torch.Tensor | None = None
        self.first_frame_mask: torch.Tensor | None = None
        self._prepare_latents = pipe.prepare_latents
        self._scheduler_step = pipe.scheduler.step

    def install(self) -> None:
        def prepare_latents(*args: Any, **kwargs: Any):
            outputs = self._prepare_latents(*args, **kwargs)
            if len(outputs) == 3:
                # Keep the recorder's copies on CPU. Allocating a full x0 tensor
                # on CUDA before the real scheduler step changes allocator
                # pressure and can perturb otherwise bit-exact fused kernels.
                _, condition, first_frame_mask = outputs
                self.condition = condition.detach().cpu().to(torch.float32)
                self.first_frame_mask = first_frame_mask.detach().cpu().to(torch.float32)
            return outputs

        def scheduler_step(model_output: torch.Tensor, timestep: Any, sample: torch.Tensor, *args: Any, **kwargs: Any):
            scheduler = self.pipe.scheduler
            prediction_type = getattr(scheduler.config, "prediction_type", "flow_prediction")
            if prediction_type != "flow_prediction":
                raise ValueError(f"Step renderer requires flow_prediction, got {prediction_type!r}")

            # Execute the untouched solver first. Only after it has produced the
            # real next state do we copy its immutable inputs to CPU and form x0.
            # This makes the recorder observational with respect to the solver.
            output = self._scheduler_step(model_output, timestep, sample, *args, **kwargs)
            step_index = int(scheduler.step_index) - 1
            sigma = float(scheduler.sigmas[step_index].detach().cpu().item())
            pred_x0 = sample.detach().cpu().to(torch.float32) - sigma * model_output.detach().cpu().to(torch.float32)
            if self.condition is not None and self.first_frame_mask is not None:
                pred_x0 = (1.0 - self.first_frame_mask) * self.condition + self.first_frame_mask * pred_x0

            self.pred_x0.append(pred_x0)
            self.sigmas.append(sigma)
            self.timesteps.append(float(timestep.detach().cpu().item() if torch.is_tensor(timestep) else timestep))
            return output

        self.pipe.prepare_latents = prepare_latents
        self.pipe.scheduler.step = scheduler_step

    def restore(self) -> None:
        self.pipe.prepare_latents = self._prepare_latents
        self.pipe.scheduler.step = self._scheduler_step


@torch.no_grad()
def _decode_diffusers_latent(pipe: Any, latent: torch.Tensor, device: torch.device) -> np.ndarray:
    latents = latent.to(device=device, dtype=pipe.vae.dtype)
    mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device=latents.device, dtype=latents.dtype)
    )
    inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        device=latents.device, dtype=latents.dtype
    )
    decoded = pipe.vae.decode(latents / inv_std + mean, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(decoded, output_type="np")[0]


def _as_uint8(video: np.ndarray) -> np.ndarray:
    return (np.clip(video, 0.0, 1.0) * 255).astype(np.uint8)


def _formal_final_frames(args: argparse.Namespace) -> np.ndarray | None:
    if args.formal_final_video is None:
        return None
    path = Path(args.formal_final_video)
    reason = eval_i2v._video_validation_error(
        path,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
    )
    if reason is not None:
        raise RuntimeError(f"Formal final video is invalid ({reason}): {path}")
    reader = decord.VideoReader(str(path), num_threads=1)
    return reader.get_batch(range(args.num_frames)).asnumpy()


def _bind_formal_final(args: argparse.Namespace, output_dir: Path) -> dict[str, Any] | None:
    if args.formal_final_video is None:
        return None
    source = Path(args.formal_final_video).resolve()
    final_path = output_dir / "final_00.mp4"
    final_step_path = output_dir / f"step_{args.num_inference_steps - 1:02d}.mp4"
    shutil.copyfile(source, final_path)
    shutil.copyfile(source, final_step_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "source": str(source),
        "sha256": digest,
        "bound_outputs": [str(final_path), str(final_step_path)],
    }


def _render_ode(
    args: argparse.Namespace,
    *,
    image: Image.Image,
    prompt: str,
    name: str,
    device: torch.device,
) -> dict[str, Any]:
    load_args = SimpleNamespace(
        model_path=args.model_path,
        checkpoint=None,
        use_ema=False,
        disable_progress_bar=True,
    )
    pipe = eval_i2v._load_pipeline(load_args, device, rank=0)
    if args.sampler == "euler":
        install_flowmatch_euler_scheduler(pipe)

    recorder = _OdeStepRecorder(pipe)
    recorder.install()
    try:
        generator = torch.Generator(device=device).manual_seed(args.seed + args.sample_index)
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
    finally:
        recorder.restore()

    if len(recorder.pred_x0) != args.num_inference_steps:
        raise RuntimeError(f"Captured {len(recorder.pred_x0)} ODE previews, expected {args.num_inference_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_final_video = np.asarray(output.frames[0])
    formal_final_video = _formal_final_frames(args)
    final_display_video = formal_final_video if formal_final_video is not None else _as_uint8(generated_final_video)
    final_path = output_dir / "final_00.mp4"
    if formal_final_video is None:
        export_to_video(list(generated_final_video), str(final_path), fps=args.fps)
    else:
        shutil.copyfile(Path(args.formal_final_video), final_path)

    videos: list[np.ndarray] = []
    previews: list[dict[str, Any]] = []
    labels: list[str] = []
    for index, pred_x0 in enumerate(recorder.pred_x0):
        is_final = index == args.num_inference_steps - 1
        step_path = output_dir / f"step_{index:02d}.mp4"
        if is_final:
            video_uint8 = final_display_video
            if formal_final_video is None:
                export_to_video(list(generated_final_video), str(step_path), fps=args.fps)
            else:
                shutil.copyfile(Path(args.formal_final_video), step_path)
        else:
            video = _decode_diffusers_latent(pipe, pred_x0, device)
            export_to_video(list(video), str(step_path), fps=args.fps)
            video_uint8 = _as_uint8(video)
        videos.append(video_uint8)
        label = (
            f"{index + 1:02d}/{args.num_inference_steps:02d} final s=0"
            if is_final
            else f"{index + 1:02d}/{args.num_inference_steps:02d} x0 s={recorder.sigmas[index]:.3f}"
        )
        labels.append(label)
        previews.append(
            {
                "display_step": index + 1,
                "file_index": index,
                "kind": "final_latent" if is_final else "predicted_clean_x0",
                "source_sigma": recorder.sigmas[index],
                "source_timestep": recorder.timesteps[index],
                "output_sigma": 0.0,
                "file": str(step_path),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grid_path = output_dir / "steps_grid.mp4"
    contact_path = output_dir / "step_contact_sheet.jpg"
    save_step_grid_video(
        videos,
        grid_path,
        fps=args.fps,
        cols=args.grid_cols,
        thumb_width=args.grid_thumb_width,
        step_labels=labels,
    )
    save_step_contact_sheet(videos, contact_path, step_labels=labels)

    manifest: dict[str, Any] = {
        "model_path": str(Path(args.model_path).resolve()),
        "eval_json": str(Path(args.eval_json).resolve()),
        "sample_index": args.sample_index,
        "sample_name": name,
        "sampler": "flowmatch_euler" if args.sampler == "euler" else "unipc",
        "sampler_implementation": (
            "src.cli.eval_i2v_euler / FlowMatchEulerDiscreteScheduler"
            if args.sampler == "euler"
            else "src.cli.eval_i2v / UniPCMultistepScheduler"
        ),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed + args.sample_index,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "sigmas": [*recorder.sigmas, 0.0],
        "timesteps": recorder.timesteps,
        "preview_contract": "post-CFG x0=x_t-sigma*v; final cell is actual solver latent at sigma=0",
        "step_preview_semantics": (
            "Steps 1..T-1 decode scheduler clean endpoints before the real solver step; step T decodes the actual "
            "final solver latent. TI2V-5B frame zero is pinned with the pipeline condition/mask in every preview."
        ),
        "step_previews": previews,
        "outputs": {
            "steps": [entry["file"] for entry in previews],
            "finals": [str(final_path)],
        },
        "grid": str(grid_path),
        "contact_sheet": str(contact_path),
        "formal_final_binding": (
            {
                "source": str(Path(args.formal_final_video).resolve()),
                "sha256": hashlib.sha256(Path(args.formal_final_video).read_bytes()).hexdigest(),
                "bound_outputs": [str(final_path), str(output_dir / f"step_{args.num_inference_steps - 1:02d}.mp4")],
            }
            if args.formal_final_video is not None
            else None
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gc.collect()
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and not args.force:
        print(f"[skip] {manifest_path} already exists (use --force to overwrite)", flush=True)
        return 0

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    item, image, prompt, name = _load_sample(args)
    print(
        f"[render] sampler={args.sampler} sample={args.sample_index}:{name} "
        f"steps={args.num_inference_steps} cfg={args.guidance_scale} seed={args.seed + args.sample_index}",
        flush=True,
    )
    started = time.time()
    if args.sampler == "cps":
        manifest = _render_cps(
            args,
            item=item,
            image=image,
            prompt=prompt,
            name=name,
            device=device,
        )
    else:
        manifest = _render_ode(args, image=image, prompt=prompt, name=name, device=device)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "grid": manifest.get("grid"),
                "contact_sheet": manifest.get("contact_sheet"),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
