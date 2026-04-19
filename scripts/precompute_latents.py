"""Precompute VAE latents and T5 embeddings, saving results to parquet.

Single-GPU:
    uv run python scripts/precompute_latents.py \
        --input data/train_maze_bfs_sft.json \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --output_dir /path/to/output \
        --batch_size 4

Multi-GPU (8 GPUs):
    uv run torchrun --nproc_per_node=8 scripts/precompute_latents.py \
        --input data/train_maze_bfs_sft.json \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --output_dir /path/to/output \
        --batch_size 4

Each GPU processes its own shard of rows and writes a separate parquet file.
Rank 0 merges all shards into the final parquet + config JSON at the end.

Input: the same JSON config that I2VDataset expects (pointing to parquet + video root).
Output: one parquet per input parquet, placed in --output_dir, containing:
    - prompt               (string)
    - prompt_embeds        (bytes, bf16 tensor)
    - video_latents_0      (bytes, bf16 tensor)  — step 0
    - video_latents_1      (bytes, bf16 tensor)  — step 1
    - ...
    - final_latents        (bytes, bf16 tensor)  — final step (alias for convenience)
    - condition            (bytes, bf16 tensor)
    - num_steps            (int)
    - latent_shape         (string, JSON list)   — shape of each video_latents_*
    - condition_shape      (string, JSON list)
    - embed_shape          (string, JSON list)

A companion JSON config is also written so you can use it directly for training.
"""

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return "RANK" in os.environ


def _get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _init_distributed():
    if not _is_distributed():
        return
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def _barrier():
    if _is_distributed():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Precompute VAE + T5 embeddings to parquet")
    p.add_argument("--input", required=True, help="Dataset JSON config (same format as training)")
    p.add_argument("--model_path", required=True, help="Path to Wan2.2 diffusers model")
    p.add_argument("--output_dir", required=True, help="Directory to write output parquet files")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for encoding")
    p.add_argument("--num_frames", type=int, default=None, help="Override num_frames")
    p.add_argument("--max_area", type=int, default=None, help="Override max_area")
    p.add_argument("--height", type=int, default=None, help="Override height")
    p.add_argument("--width", type=int, default=None, help="Override width")
    p.add_argument("--fps", type=int, default=None, help="Override fps")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tensor serialization
# ---------------------------------------------------------------------------


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().cpu().numpy().tobytes()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_components(model_path: str, device: str):
    """Load only VAE + T5 (no transformers needed)."""
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    model_dir = Path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    logger.info("Loaded tokenizer")

    text_encoder = UMT5EncoderModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
    text_encoder.to(device).eval().requires_grad_(False)
    logger.info("Loaded text encoder")

    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32)
    vae.to(device).eval().requires_grad_(False)
    logger.info("Loaded VAE")

    vae_cfg = vae.config
    latents_mean = torch.tensor(vae_cfg.latents_mean).view(1, vae_cfg.z_dim, 1, 1, 1).to(device)
    latents_std_inv = (1.0 / torch.tensor(vae_cfg.latents_std)).view(1, vae_cfg.z_dim, 1, 1, 1).to(device)
    scale_spatial = vae_cfg.scale_factor_spatial
    scale_temporal = vae_cfg.scale_factor_temporal

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "latents_mean": latents_mean,
        "latents_std_inv": latents_std_inv,
        "scale_spatial": scale_spatial,
        "scale_temporal": scale_temporal,
    }


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_text(components, prompts: list[str], device: str) -> torch.Tensor:
    """Encode prompts -> (B, 512, text_dim) bf16."""
    import html

    import ftfy
    import regex as re

    def clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    prompts = [clean(p) for p in prompts]
    tokens = components["tokenizer"](
        prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    embeds = components["text_encoder"](input_ids, mask).last_hidden_state
    embeds = embeds.masked_fill(~mask.bool().unsqueeze(-1), 0)
    return embeds.to(torch.bfloat16)


@torch.no_grad()
def encode_video(components, video: torch.Tensor) -> torch.Tensor:
    """Encode pixel video (B, C, T, H, W) in [-1, 1] -> normalized latents bf16."""
    vae = components["vae"]
    latents = vae.encode(video.to(vae.dtype)).latent_dist.mode()
    mean = components["latents_mean"].to(dtype=latents.dtype)
    std_inv = components["latents_std_inv"].to(dtype=latents.dtype)
    return ((latents - mean) * std_inv).to(torch.bfloat16)


@torch.no_grad()
def prepare_condition(components, image: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Build condition tensor (B, 4+z_dim, T', H', W') bf16."""
    vae = components["vae"]
    scale_spatial = components["scale_spatial"]
    scale_temporal = components["scale_temporal"]

    B = image.shape[0]
    cond_video = image.new_zeros((B, 3, num_frames, height, width))
    cond_video[:, :, 0] = image

    cond_latents = vae.encode(cond_video.to(vae.dtype)).latent_dist.mode()
    mean = components["latents_mean"].to(dtype=cond_latents.dtype)
    std_inv = components["latents_std_inv"].to(dtype=cond_latents.dtype)
    cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)

    latent_h = height // scale_spatial
    latent_w = width // scale_spatial
    mask = torch.ones(1, 1, num_frames, latent_h, latent_w, device=image.device, dtype=cond_latents.dtype)
    mask[:, :, 1:] = 0
    first_frame_mask = mask[:, :, 0:1].repeat(1, 1, scale_temporal, 1, 1)
    mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
    mask = mask.view(1, -1, scale_temporal, latent_h, latent_w).transpose(1, 2).contiguous()
    mask = mask.expand(B, -1, -1, -1, -1)

    return torch.cat([mask, cond_latents], dim=1)


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------


def load_video(video_path: str, height: int, width: int, num_frames: int) -> torch.Tensor:
    """Load video frames as uint8 (C, T, H, W)."""
    import decord
    import numpy as np

    vr = decord.VideoReader(video_path, width=width, height=height)
    indices = np.linspace(0, len(vr) - 1, num_frames).round().astype(int).tolist()
    frames = vr.get_batch(indices)  # (T, H, W, C)
    return frames.permute(3, 0, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_rows(
    table: pa.Table,
    row_indices: list[int],
    root: Path,
    components: dict,
    device: str,
    batch_size: int,
    cfg_num_frames: int,
    cfg_max_area: int,
    cfg_height: int | None,
    cfg_width: int | None,
) -> pa.Table:
    """Process a subset of rows from a table, return a new table with precomputed latents."""
    import decord
    import numpy as np

    from src.data.i2v_dataset import compute_hw

    decord.bridge.set_bridge("torch")
    cols = table.column_names
    n = len(row_indices)

    # Determine number of steps from the first row
    if "videos" in cols:
        num_steps = len(table.column("videos")[row_indices[0]].as_py())
    elif "video" in cols:
        num_steps = 1
    else:
        raise ValueError("No 'videos' or 'video' column")

    # Accumulators
    prompts_col: list[str] = []
    prompt_embeds_col: list[bytes] = []
    step_latents_cols: list[list[bytes]] = [[] for _ in range(num_steps)]
    condition_col: list[bytes] = []
    latent_shape_col: list[str] = []
    condition_shape_col: list[str] = []
    embed_shape_col: list[str] = []

    def resolve(p: str) -> str:
        pp = Path(p)
        return str(pp) if pp.is_absolute() else str(root / pp)

    for batch_start in tqdm(range(0, n, batch_size), desc=f"Encoding (GPU {_get_rank()})", disable=_get_rank() != 0):
        batch_end = min(batch_start + batch_size, n)
        batch_rows = row_indices[batch_start:batch_end]
        cur_batch_size = len(batch_rows)
        batch_prompts: list[str] = []
        batch_all_videos: list[list[torch.Tensor]] = [[] for _ in range(num_steps)]
        batch_images: list[torch.Tensor] = []
        h, w = 0, 0

        for i in batch_rows:
            video_paths = table.column("videos")[i].as_py() if "videos" in cols else [table.column("video")[i].as_py()]

            prompt = table.column("prompt")[i].as_py() if "prompt" in cols else ""
            image_path = table.column("image")[i].as_py() if "image" in cols else None

            final_video = resolve(video_paths[-1])
            if cfg_height is not None and cfg_width is not None:
                h, w = cfg_height, cfg_width
            else:
                vr = decord.VideoReader(final_video)
                orig_h, orig_w = vr[0].shape[:2]
                h, w = compute_hw(cfg_max_area, orig_h / orig_w)

            for step_idx, vp in enumerate(video_paths):
                video_tensor = load_video(resolve(vp), h, w, cfg_num_frames)
                batch_all_videos[step_idx].append(video_tensor)

            if image_path is not None:
                from PIL import Image

                with Image.open(resolve(image_path)) as img:
                    img = img.convert("RGB").resize((w, h), Image.LANCZOS)
                    image_tensor = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()
            else:
                image_tensor = batch_all_videos[-1][-1][:, 0].clone()

            batch_prompts.append(prompt)
            batch_images.append(image_tensor)

        # Encode text
        prompt_emb = encode_text(components, batch_prompts, device)

        # Encode each step's videos
        step_latents: list[torch.Tensor] = []
        for step_idx in range(num_steps):
            video_batch = (
                torch.stack(batch_all_videos[step_idx]).to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            )
            step_latents.append(encode_video(components, video_batch))

        # Encode condition
        image_batch = torch.stack(batch_images).to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
        cond = prepare_condition(components, image_batch, cfg_num_frames, h, w)

        # Store per-sample
        for j in range(cur_batch_size):
            prompts_col.append(batch_prompts[j])
            prompt_embeds_col.append(tensor_to_bytes(prompt_emb[j]))
            embed_shape_col.append(json.dumps(list(prompt_emb[j].shape)))

            for step_idx in range(num_steps):
                step_latents_cols[step_idx].append(tensor_to_bytes(step_latents[step_idx][j]))

            latent_shape_col.append(json.dumps(list(step_latents[0][j].shape)))
            condition_col.append(tensor_to_bytes(cond[j]))
            condition_shape_col.append(json.dumps(list(cond[j].shape)))

    out_columns = {
        "prompt": prompts_col,
        "prompt_embeds": prompt_embeds_col,
    }
    for step_idx in range(num_steps - 1):
        out_columns[f"video_latents_{step_idx}"] = step_latents_cols[step_idx]
    out_columns["final_latents"] = step_latents_cols[-1]
    out_columns.update(
        {
            "condition": condition_col,
            "num_steps": [num_steps] * n,
            "embed_shape": embed_shape_col,
            "latent_shape": latent_shape_col,
            "condition_shape": condition_shape_col,
        }
    )
    return pa.table(out_columns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    _init_distributed()

    rank = _get_rank()
    world_size = _get_world_size()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(input_path.read_text())
    entries = [raw] if isinstance(raw, dict) else raw

    if rank == 0:
        logger.info("Loading model components from {} (world_size={})", args.model_path, world_size)
    components = load_model_components(args.model_path, device)

    output_entries = []

    for entry in entries:
        data_path = Path(entry["data_path"])
        if not data_path.is_absolute():
            data_path = input_path.parent / data_path

        root = Path(entry.get("root", data_path.parent))
        if not root.is_absolute():
            root = input_path.parent / root

        cfg_num_frames = args.num_frames or entry.get("num_frames", 81)
        cfg_max_area = args.max_area or entry.get("max_area", 480 * 832)
        cfg_height = args.height or entry.get("height")
        cfg_width = args.width or entry.get("width")

        table = pq.read_table(data_path)
        n = table.num_rows
        if rank == 0:
            logger.info("Processing {} — {} rows across {} GPUs", data_path, n, world_size)

        # Shard rows across GPUs
        all_indices = list(range(n))
        my_indices = all_indices[rank::world_size]

        if rank == 0:
            logger.info("  Rank 0 processing {} rows", len(my_indices))

        shard_table = process_rows(
            table,
            my_indices,
            root,
            components,
            device,
            args.batch_size,
            cfg_num_frames,
            cfg_max_area,
            cfg_height,
            cfg_width,
        )

        # Write per-rank shard
        shard_path = output_dir / f"{data_path.stem}_latents_shard{rank}.parquet"
        pq.write_table(shard_table, shard_path)
        if rank == 0:
            logger.info("  Rank 0 shard written to {}", shard_path)

        # Wait for all ranks to finish writing
        _barrier()

        # Rank 0 merges all shards
        if rank == 0:
            shard_tables = []
            for r in range(world_size):
                sp = output_dir / f"{data_path.stem}_latents_shard{r}.parquet"
                shard_tables.append(pq.read_table(sp))

            merged = pa.concat_tables(shard_tables)
            out_path = output_dir / f"{data_path.stem}_latents.parquet"
            pq.write_table(merged, out_path)
            logger.info("  Merged {} shards -> {} ({} rows)", world_size, out_path, merged.num_rows)

            # Clean up shards
            for r in range(world_size):
                sp = output_dir / f"{data_path.stem}_latents_shard{r}.parquet"
                sp.unlink()

            output_entries.append({"data_path": str(out_path)})

        _barrier()

    # Write companion config JSON (rank 0 only)
    if rank == 0:
        config_out = output_dir / (input_path.stem + "_latents.json")
        config_out.write_text(json.dumps(output_entries, indent=2))
        logger.info("Config written to {}", config_out)

    if _is_distributed():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
