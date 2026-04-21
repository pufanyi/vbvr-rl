"""Generate a synthetic maze dataset as WebDataset tar shards of precomputed latents.

End-to-end pipeline (runnable under torchrun):

    1. Each rank owns a disjoint slice of tar shards.
    2. For every sample, the rank synthesises a maze layout + BFS path and
       renders an RGB video (CPU).
    3. The rank batch-encodes video + first-frame condition through the
       Wan2.2 VAE and the prompt through the UMT5 text encoder (GPU).
    4. Results are packed into ``shard-NNNNNN.tar`` files whose sample
       layout matches ``VBVRLatentDataset``:

           {key:07d}.safetensors   → prompt_embeds / latents / condition
                                     plus maze_* tensors for reward use
           {key:07d}.json          → prompt / tar / index_in_tar / seq_len
                                     plus a structured ``maze`` blob

The extra ``maze_*`` tensors and JSON fields are ignored by
``VBVRLatentDataset._decode_sample`` — they are only consumed by a future
maze-aware dataset / reward function.

Launch (single node, 8 GPUs)::

    UV_NO_SYNC=1 uv run torchrun --nproc_per_node=8 \
        -m src.precompute.maze_webdataset \
        --output_dir data/maze_synth/latents/webdataset \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --num_samples 20000 --samples_per_shard 500

The output directory is compatible with ``configs/train_correction_vbvr.yaml``
by setting ``latent_webdataset_dir: data/maze_synth/latents/webdataset``.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import os
import tarfile
from pathlib import Path

import ftfy
import numpy as np
import regex as re
import torch
from loguru import logger
from pydantic import BaseModel, Field
from safetensors.torch import save as st_save
from tqdm import tqdm

from src.precompute.maze_generator import (
    DEFAULT_PALETTES,
    MazeSample,
    MazeSpec,
    build_maze_sample,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GenConfig(BaseModel):
    """Top-level config for maze dataset generation."""

    # Output
    output_dir: str
    tar_tag: str = "maze_synthetic"

    # Volume
    num_samples: int = Field(ge=1)
    samples_per_shard: int = Field(default=500, ge=1)

    # Model
    model_path: str

    # Maze geometry
    cell_h: int = Field(default=4, ge=2)
    cell_w: int = Field(default=4, ge=2)
    cell_px: int = Field(default=32, ge=4)  # image H = (2*cell_h+1)*cell_px, must land on mult of 8
    num_frames: int = Field(default=81, ge=2)

    # Encoding
    vae_batch_size: int = Field(default=2, ge=1)
    text_batch_size: int = Field(default=32, ge=1)

    # Reproducibility
    seed: int = 42

    # Misc
    skip_existing: bool = False

    def maze_spec(self) -> MazeSpec:
        return MazeSpec(
            cell_h=self.cell_h,
            cell_w=self.cell_w,
            cell_px=self.cell_px,
            num_frames=self.num_frames,
            palettes=list(DEFAULT_PALETTES),
        )


# ---------------------------------------------------------------------------
# CLI parsing (auto-generated from GenConfig)
# ---------------------------------------------------------------------------


def parse_args() -> GenConfig:
    p = argparse.ArgumentParser(description="Generate maze WebDataset with precomputed latents")
    for name, field in GenConfig.model_fields.items():
        flag = f"--{name}"
        ann = field.annotation
        if ann is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif ann is int:
            p.add_argument(flag, type=int, default=None)
        elif ann is float:
            p.add_argument(flag, type=float, default=None)
        else:
            p.add_argument(flag, type=str, default=None)
    args = p.parse_args()
    raw = {name: getattr(args, name) for name in GenConfig.model_fields if getattr(args, name) is not None}
    return GenConfig(**raw)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return "RANK" in os.environ


def _get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _init_distributed() -> None:
    if not _is_distributed():
        return
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


# ---------------------------------------------------------------------------
# Resolution sanity
# ---------------------------------------------------------------------------


_SPATIAL_DIVISOR = 8  # Wan2.2 VAE scale_factor_spatial


def _validate_resolution(cfg: GenConfig) -> tuple[int, int]:
    h = (2 * cfg.cell_h + 1) * cfg.cell_px
    w = (2 * cfg.cell_w + 1) * cfg.cell_px
    if h % _SPATIAL_DIVISOR != 0 or w % _SPATIAL_DIVISOR != 0:
        raise ValueError(
            f"Maze image resolution {h}x{w} must be divisible by {_SPATIAL_DIVISOR}. "
            f"Pick cell_px / cell counts that satisfy (2*cells+1)*cell_px % 8 == 0."
        )
    return h, w


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_vae(model_path: str, device: str) -> dict:
    from diffusers import AutoencoderKLWan

    model_dir = Path(model_path)
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32)
    vae.to(device).eval().requires_grad_(False)
    vae_cfg = vae.config
    mean = torch.tensor(vae_cfg.latents_mean).view(1, vae_cfg.z_dim, 1, 1, 1).to(device)
    std_inv = (1.0 / torch.tensor(vae_cfg.latents_std)).view(1, vae_cfg.z_dim, 1, 1, 1).to(device)
    return {
        "vae": vae,
        "latents_mean": mean,
        "latents_std_inv": std_inv,
        "scale_spatial": vae_cfg.scale_factor_spatial,
        "scale_temporal": vae_cfg.scale_factor_temporal,
    }


def _load_text_encoder(model_path: str, device: str) -> dict:
    from transformers import AutoTokenizer, UMT5EncoderModel

    model_dir = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
    text_encoder.to(device).eval().requires_grad_(False)
    max_length = tokenizer.model_max_length
    if max_length is None or max_length > 10_000:
        max_length = 512
    return {"tokenizer": tokenizer, "text_encoder": text_encoder, "max_length": max_length}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encode_video(vae_bundle: dict, video_bf16_m1p1: torch.Tensor) -> torch.Tensor:
    vae = vae_bundle["vae"]
    latents = vae.encode(video_bf16_m1p1.to(vae.dtype)).latent_dist.mode()
    mean = vae_bundle["latents_mean"].to(dtype=latents.dtype)
    std_inv = vae_bundle["latents_std_inv"].to(dtype=latents.dtype)
    return ((latents - mean) * std_inv).to(torch.bfloat16)


@torch.no_grad()
def _encode_condition(
    vae_bundle: dict,
    first_frame_bf16_m1p1: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    vae = vae_bundle["vae"]
    scale_spatial = vae_bundle["scale_spatial"]
    scale_temporal = vae_bundle["scale_temporal"]

    B = first_frame_bf16_m1p1.shape[0]
    cond_video = first_frame_bf16_m1p1.new_zeros((B, 3, num_frames, height, width))
    cond_video[:, :, 0] = first_frame_bf16_m1p1

    cond_latents = vae.encode(cond_video.to(vae.dtype)).latent_dist.mode()
    mean = vae_bundle["latents_mean"].to(dtype=cond_latents.dtype)
    std_inv = vae_bundle["latents_std_inv"].to(dtype=cond_latents.dtype)
    cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)

    latent_h = height // scale_spatial
    latent_w = width // scale_spatial
    mask = torch.ones(1, 1, num_frames, latent_h, latent_w, device=cond_latents.device, dtype=cond_latents.dtype)
    mask[:, :, 1:] = 0
    first_frame_mask = mask[:, :, 0:1].repeat(1, 1, scale_temporal, 1, 1)
    mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
    mask = mask.view(1, -1, scale_temporal, latent_h, latent_w).transpose(1, 2).contiguous()
    mask = mask.expand(B, -1, -1, -1, -1)
    return torch.cat([mask, cond_latents], dim=1)


@torch.no_grad()
def _encode_prompts(text_bundle: dict, prompts: list[str], device: str) -> list[torch.Tensor]:
    def _clean(t: str) -> str:
        t = ftfy.fix_text(t)
        t = html.unescape(html.unescape(t))
        t = re.sub(r"\s+", " ", t).strip()
        return t

    cleaned = [_clean(p) for p in prompts]
    tokens = text_bundle["tokenizer"](
        cleaned,
        padding="longest",
        max_length=text_bundle["max_length"],
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    embeds = text_bundle["text_encoder"](input_ids, mask).last_hidden_state.to(torch.bfloat16)
    seq_lens = mask.sum(dim=1).tolist()
    return [embeds[i, :length].contiguous() for i, length in enumerate(seq_lens)]


# ---------------------------------------------------------------------------
# Sample conversion helpers
# ---------------------------------------------------------------------------


def _video_to_tensor(video_uint8: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (3, T, H, W) float32 in [-1, 1]."""
    t = torch.from_numpy(video_uint8).permute(3, 0, 1, 2).contiguous()  # (3, T, H, W) uint8
    return t.to(torch.float32).div_(127.5).sub_(1.0)


def _sample_to_reward_tensors(sample: MazeSample) -> dict[str, torch.Tensor]:
    """Small tensor payload for fast reward-side reads. All CPU, no bf16.

    ``maze_path`` is intentionally **excluded**: its length varies per sample
    and would break default DataLoader collation.  The BFS path is still
    available in the JSON blob if needed offline.  Fixed-shape tensors only
    in the safetensors pass-through.
    """
    return {
        "maze_grid": torch.tensor(sample.grid, dtype=torch.int8),
        "maze_frame_positions_cell": torch.tensor(sample.frame_positions_cell, dtype=torch.float32),
        "maze_frame_positions_pix": torch.tensor(sample.frame_positions_pix, dtype=torch.float32),
        "maze_start": torch.tensor(sample.start, dtype=torch.int16),
        "maze_goal": torch.tensor(sample.goal, dtype=torch.int16),
        "maze_ball_rgb": torch.tensor(sample.palette.ball_rgb, dtype=torch.uint8),
        "maze_goal_rgb": torch.tensor(sample.palette.goal_rgb, dtype=torch.uint8),
        "maze_wall_rgb": torch.tensor(sample.palette.wall_rgb, dtype=torch.uint8),
        "maze_passage_rgb": torch.tensor(sample.palette.passage_rgb, dtype=torch.uint8),
        "maze_cell_px": torch.tensor(sample.cell_px, dtype=torch.int32),
        "maze_image_hw": torch.tensor([sample.image_h, sample.image_w], dtype=torch.int32),
    }


def _sample_to_json_blob(sample: MazeSample) -> dict:
    """JSON-friendly maze record — everything human-readable for offline use."""
    return {
        "cell_h": sample.cell_h,
        "cell_w": sample.cell_w,
        "cell_px": sample.cell_px,
        "image_h": sample.image_h,
        "image_w": sample.image_w,
        "num_frames": sample.num_frames,
        "start": list(sample.start),
        "goal": list(sample.goal),
        "path_len": sample.path_len,
        "path": [list(p) for p in sample.path],
        "frame_positions_cell": [list(p) for p in sample.frame_positions_cell],
        "frame_positions_pix": [list(p) for p in sample.frame_positions_pix],
        "grid": sample.grid,
        "palette": {
            "wall_rgb": list(sample.palette.wall_rgb),
            "passage_rgb": list(sample.palette.passage_rgb),
            "ball_rgb": list(sample.palette.ball_rgb),
            "goal_rgb": list(sample.palette.goal_rgb),
            "wall_name": sample.palette.wall_name,
            "passage_name": sample.palette.passage_name,
            "ball_name": sample.palette.ball_name,
            "goal_name": sample.palette.goal_name,
        },
    }


# ---------------------------------------------------------------------------
# Tar writing
# ---------------------------------------------------------------------------


def _tar_add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------


def _shard_plan(cfg: GenConfig) -> list[tuple[int, int, int]]:
    """Global plan: (shard_id, start_global_idx, count)."""
    plan = []
    for sid in range((cfg.num_samples + cfg.samples_per_shard - 1) // cfg.samples_per_shard):
        start = sid * cfg.samples_per_shard
        count = min(cfg.samples_per_shard, cfg.num_samples - start)
        plan.append((sid, start, count))
    return plan


def _generate_shard_samples(
    cfg: GenConfig,
    spec: MazeSpec,
    start_global_idx: int,
    count: int,
) -> list[tuple[np.ndarray, MazeSample, int]]:
    """CPU-only: synthesise ``count`` maze samples with deterministic seeds."""
    out = []
    for k in range(count):
        gid = start_global_idx + k
        rng = np.random.default_rng(cfg.seed + gid)
        video, sample = build_maze_sample(spec, rng)
        out.append((video, sample, gid))
    return out


def _encode_shard(
    cfg: GenConfig,
    samples: list[tuple[np.ndarray, MazeSample, int]],
    vae_bundle: dict,
    text_bundle: dict,
    device: str,
    height: int,
    width: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, MazeSample, int]]:
    """Run VAE + text encoder over a shard's samples. Returns per-sample tensors (CPU)."""
    n = len(samples)
    latents_all: list[torch.Tensor] = [None] * n  # type: ignore[list-item]
    condition_all: list[torch.Tensor] = [None] * n  # type: ignore[list-item]

    # Video + condition (VAE) in small batches.
    for batch_start in range(0, n, cfg.vae_batch_size):
        batch = samples[batch_start : batch_start + cfg.vae_batch_size]
        vids = torch.stack([_video_to_tensor(v) for v, _, _ in batch]).to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        first_frames = vids[:, :, 0].contiguous()
        latents = _encode_video(vae_bundle, vids)
        cond = _encode_condition(vae_bundle, first_frames, cfg.num_frames, height, width)
        for j in range(len(batch)):
            latents_all[batch_start + j] = latents[j].contiguous().cpu()
            condition_all[batch_start + j] = cond[j].contiguous().cpu()

    # Prompts (text encoder) in larger batches.
    prompt_embeds_all: list[torch.Tensor] = [None] * n  # type: ignore[list-item]
    for batch_start in range(0, n, cfg.text_batch_size):
        batch = samples[batch_start : batch_start + cfg.text_batch_size]
        prompts = [s.prompt for _, s, _ in batch]
        embeds = _encode_prompts(text_bundle, prompts, device)
        for j, e in enumerate(embeds):
            prompt_embeds_all[batch_start + j] = e.cpu()

    return [(latents_all[i], condition_all[i], prompt_embeds_all[i], samples[i][1], samples[i][2]) for i in range(n)]


def _write_shard(
    cfg: GenConfig,
    shard_id: int,
    encoded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, MazeSample, int]],
    output_dir: Path,
) -> Path:
    """Pack encoded samples into a single shard-NNNNNN.tar (atomic)."""
    final_path = output_dir / f"shard-{shard_id:06d}.tar"
    tmp_path = final_path.with_suffix(".tar.tmp")
    with tarfile.open(tmp_path, "w") as tar:
        for latents, condition, prompt_embeds, sample, gid in encoded:
            key = f"{gid:07d}"

            st_tensors = {
                "prompt_embeds": prompt_embeds,
                "latents": latents,
                "condition": condition,
                **_sample_to_reward_tensors(sample),
            }
            st_bytes = st_save(st_tensors)

            meta = {
                "prompt": sample.prompt,
                "tar": cfg.tar_tag,
                "index_in_tar": gid,
                "seq_len": int(prompt_embeds.shape[0]),
                "maze": _sample_to_json_blob(sample),
            }
            meta_bytes = json.dumps(meta).encode()

            _tar_add_bytes(tar, f"{key}.safetensors", st_bytes)
            _tar_add_bytes(tar, f"{key}.json", meta_bytes)
    tmp_path.rename(final_path)
    return final_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    _init_distributed()

    rank = _get_rank()
    world_size = _get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = _validate_resolution(cfg)
    spec = cfg.maze_spec()

    if rank == 0:
        logger.info(
            "Maze gen: {} samples @ {}x{} px, {} frames, cells {}x{}, cell_px {}",
            cfg.num_samples,
            height,
            width,
            cfg.num_frames,
            cfg.cell_h,
            cfg.cell_w,
            cfg.cell_px,
        )

    plan = _shard_plan(cfg)
    my_shards = plan[rank::world_size]
    if cfg.skip_existing:
        my_shards = [s for s in my_shards if not (output_dir / f"shard-{s[0]:06d}.tar").exists()]
    logger.info("[rank {}] assigned {} / {} shards", rank, len(my_shards), len(plan))

    torch.backends.cudnn.benchmark = True
    logger.info("[rank {}] loading VAE", rank)
    vae_bundle = _load_vae(cfg.model_path, device)
    logger.info("[rank {}] loading text encoder", rank)
    text_bundle = _load_text_encoder(cfg.model_path, device)

    pbar = tqdm(total=len(my_shards), desc=f"[rank {rank}]", position=local_rank, leave=True, unit="shard")
    for shard_id, start_gid, count in my_shards:
        samples = _generate_shard_samples(cfg, spec, start_gid, count)
        encoded = _encode_shard(cfg, samples, vae_bundle, text_bundle, device, height, width)
        _write_shard(cfg, shard_id, encoded, output_dir)
        pbar.update(1)
    pbar.close()

    # Rank 0 writes a small dataset_info.json for downstream tooling.
    if rank == 0:
        info = {
            "tar_tag": cfg.tar_tag,
            "num_samples": cfg.num_samples,
            "samples_per_shard": cfg.samples_per_shard,
            "num_shards": len(plan),
            "image_h": height,
            "image_w": width,
            "num_frames": cfg.num_frames,
            "cell_h": cfg.cell_h,
            "cell_w": cfg.cell_w,
            "cell_px": cfg.cell_px,
            "seed": cfg.seed,
            "model_path": cfg.model_path,
        }
        (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))

    if _is_distributed():
        with contextlib.suppress(Exception):
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    logger.info("[rank {}] done", rank)


if __name__ == "__main__":
    main()
