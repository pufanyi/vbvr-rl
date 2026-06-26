"""Decode latents to video and write the run's artifacts.

Canonical home for the small, generic media helpers that were previously copied
across the sampling CLIs: latent -> uint8 video, the per-step z0 contact sheet,
and the per-step grid video. :func:`write_outputs` ties them together with a
final mp4 per batch member and a JSON manifest.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.models.wan_i2v import WanI2VForTraining

from .config import InferenceConfig
from .engine import StepwiseResult
from .inputs import PreparedInput


# ----------------------------------------------------------------------
# Latent -> uint8 video
# ----------------------------------------------------------------------
def _uint8_from_decoded(decoded: torch.Tensor) -> np.ndarray:
    """(B, C, T, H, W) in [-1, 1] -> (B, T, H, W, C) uint8."""
    decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return decoded.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()


def decode_batch_to_uint8(model: WanI2VForTraining, latents: torch.Tensor) -> np.ndarray:
    """Decode a batch of latents to (B, T, H, W, C) uint8 frames."""
    with torch.no_grad():
        decoded = model.decode_latents(latents)
    return _uint8_from_decoded(decoded)


def decode_latents_to_uint8(model: WanI2VForTraining, latents: torch.Tensor) -> np.ndarray:
    """Decode latents and return the first member's (T, H, W, C) uint8 frames."""
    return decode_batch_to_uint8(model, latents)[0]


def export_uint8_video(video: np.ndarray, path: Path, fps: int) -> None:
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(frame) for frame in video]
    export_to_video(frames, str(path), fps=fps)


# ----------------------------------------------------------------------
# Per-step galleries
# ----------------------------------------------------------------------
def save_step_contact_sheet(
    videos: list[np.ndarray],
    path: Path,
    *,
    frame_count: int = 5,
    thumb_width: int = 160,
) -> None:
    """One row per step; a few evenly-spaced frames per row."""
    if not videos:
        return

    total_frames = videos[0].shape[0]
    frame_indices = (
        [total_frames - 1]
        if frame_count <= 1
        else np.linspace(0, total_frames - 1, frame_count).round().astype(int).tolist()
    )
    aspect = videos[0].shape[1] / videos[0].shape[2]
    thumb_height = max(1, int(round(thumb_width * aspect)))
    label_h = 18
    pad = 4
    cols = len(frame_indices)
    rows = len(videos)
    sheet = Image.new(
        "RGB",
        (cols * thumb_width + (cols + 1) * pad, rows * (thumb_height + label_h) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, video in enumerate(videos):
        y = pad + row * (thumb_height + label_h + pad)
        draw.text((pad, y), f"s{row:02d}", fill=(0, 0, 0))
        for col, frame_idx in enumerate(frame_indices):
            frame = Image.fromarray(video[frame_idx]).resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
            x = pad + col * (thumb_width + pad)
            sheet.paste(frame, (x, y + label_h))
            if row == 0:
                draw.text((x + 2, y + label_h + 2), f"f{frame_idx}", fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_step_grid_video(
    videos: list[np.ndarray],
    path: Path,
    *,
    fps: int,
    cols: int,
    thumb_width: int,
) -> None:
    """Tile every step's preview into one grid video (one cell per step)."""
    if not videos:
        return

    from diffusers.utils import export_to_video

    cols = max(1, cols)
    rows = int(np.ceil(len(videos) / cols))
    frame_count = min(video.shape[0] for video in videos)
    aspect = videos[0].shape[1] / videos[0].shape[2]
    thumb_height = max(1, int(round(thumb_width * aspect)))
    label_h = 18
    pad = 4
    grid_w = cols * thumb_width + (cols + 1) * pad
    grid_h = rows * (thumb_height + label_h) + (rows + 1) * pad

    frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        canvas = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(canvas)
        for step_idx, video in enumerate(videos):
            row, col = divmod(step_idx, cols)
            x = pad + col * (thumb_width + pad)
            y = pad + row * (thumb_height + label_h + pad)
            draw.text((x + 2, y), f"step {step_idx:02d}", fill=(0, 0, 0))
            frame = Image.fromarray(video[frame_idx]).resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
            canvas.paste(frame, (x, y + label_h))
        frames.append(canvas)

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=fps)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def write_outputs(
    model: WanI2VForTraining,
    cfg: InferenceConfig,
    prepared: PreparedInput,
    result: StepwiseResult,
    out_dir: Path,
    started: float | None = None,
) -> dict[str, Any]:
    """Decode and write references, per-step previews, final videos and manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    written: dict[str, list[str]] = {"references": [], "steps": [], "finals": []}

    # ---- reference videos (latent input only) ----
    if cfg.save_reference and prepared.reference_latents:
        for idx, ref in enumerate(prepared.reference_latents):
            ref_path = out_dir / f"reference_{idx}.mp4"
            ref_video = decode_latents_to_uint8(model, ref.unsqueeze(0).to(device))
            export_uint8_video(ref_video, ref_path, cfg.fps)
            written["references"].append(str(ref_path))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- per-step z0 previews (member 0) ----
    grid_path = contact_path = None
    if cfg.save_steps and result.pred_x0:
        step_videos: list[np.ndarray] = []
        for step_idx, z0 in enumerate(result.pred_x0):
            step_path = out_dir / f"step_{step_idx:02d}.mp4"
            video = decode_latents_to_uint8(model, z0.to(device))
            export_uint8_video(video, step_path, cfg.fps)
            step_videos.append(video)
            written["steps"].append(str(step_path))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        grid_path = out_dir / "steps_grid.mp4"
        contact_path = out_dir / "step_contact_sheet.jpg"
        save_step_grid_video(step_videos, grid_path, fps=cfg.fps, cols=cfg.grid_cols, thumb_width=cfg.grid_thumb_width)
        save_step_contact_sheet(step_videos, contact_path)

    # ---- final videos (all batch members) ----
    finals = decode_batch_to_uint8(model, result.final_latent.to(device))
    for member in range(finals.shape[0]):
        final_path = out_dir / f"final_{member:02d}.mp4"
        export_uint8_video(finals[member], final_path, cfg.fps)
        written["finals"].append(str(final_path))

    # ---- manifest ----
    manifest: dict[str, Any] = {
        "model_path": cfg.model_path,
        "checkpoint": cfg.checkpoint,
        "use_ema": cfg.use_ema,
        "mode": cfg.mode,
        "sde_formula": cfg.sde_formula,
        "noise_scale": cfg.effective_noise_scale,
        "num_sampling_steps": cfg.num_sampling_steps,
        "cfg_scale": cfg.cfg_scale,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "share_init_noise": cfg.share_init_noise,
        "source": prepared.source,
        "summary": prepared.summary,
        "sigmas": result.sigmas,
        "timesteps": result.timesteps,
        "fps": cfg.fps,
        "output_dir": str(out_dir),
        "outputs": written,
        "grid": str(grid_path) if grid_path is not None else None,
        "contact_sheet": str(contact_path) if contact_path is not None else None,
    }
    if started is not None:
        manifest["elapsed_seconds"] = time.time() - started
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if prepared.metadata:
        (out_dir / "sample_metadata.json").write_text(json.dumps(prepared.metadata, indent=2), encoding="utf-8")
    return manifest
