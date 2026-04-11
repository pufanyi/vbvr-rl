"""Combine prompt embeddings + VAE latents into WebDataset tar shards.

Reads:
  prompt_embeds_dir/rank*_batch*.safetensors  — T5 prompt embeddings
  vae_latents_dir/{tar_stem}_{idx}.safetensors — VAE latents + condition

Writes:
  output_dir/shard-NNNNNN.tar

Each sample in the tar has two entries:
  {key}.safetensors  — prompt_embeds, latents, condition
  {key}.json         — prompt, tar, index_in_tar, seq_len

Usage:
    uv run python -m src.precompute.build_webdataset \
        --prompt_embeds_dir data/vbvr/latents/prompt_embeds \
        --vae_latents_dir   data/vbvr/latents/vae_latents \
        --output_dir        data/vbvr/latents/webdataset \
        --samples_per_shard 1000

Training:
    import webdataset as wds
    from safetensors.torch import load as st_load

    dataset = (
        wds.WebDataset("data/vbvr/latents/webdataset/shard-{000000..NNNNNN}.tar")
        .shuffle(1000)
        .map(lambda s: {**st_load(s["safetensors"]), **json.loads(s["json"])})
    )
"""

import argparse
import io
import json
import random
import struct
import tarfile
from pathlib import Path

from loguru import logger
from safetensors import safe_open
from safetensors.torch import save as st_save
from tqdm import tqdm


def read_safetensors_header(path: str) -> dict:
    """Read only the JSON header (no tensor data)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def parse_args():
    p = argparse.ArgumentParser(description="Build WebDataset from prompt embeds + VAE latents")
    p.add_argument("--prompt_embeds_dir", required=True)
    p.add_argument("--vae_latents_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--samples_per_shard", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _add_tar_entry(tar: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def main():
    args = parse_args()
    prompt_dir = Path(args.prompt_embeds_dir)
    vae_dir = Path(args.vae_latents_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1: scan prompt embed headers (fast, no tensor data loaded)
    # ------------------------------------------------------------------
    logger.info("Scanning prompt embed headers …")
    source_files = sorted(prompt_dir.glob("*.safetensors"))
    all_samples: list[dict] = []

    for sf_path in tqdm(source_files, desc="Scanning prompt embeds"):
        header = read_safetensors_header(str(sf_path))
        meta_raw = header.pop("__metadata__", {})
        samples_meta = json.loads(meta_raw.get("samples", "[]"))

        for key, info in header.items():
            idx = int(key)
            sm = samples_meta[idx] if idx < len(samples_meta) else {}
            tar_name = sm.get("tar", "")
            all_samples.append({
                "source_file": str(sf_path),
                "key": key,
                "seq_len": info["shape"][0],
                "prompt": sm.get("prompt", ""),
                "tar": tar_name,
                "tar_stem": Path(tar_name).stem if tar_name else "",
                "index_in_tar": sm.get("index_in_tar", -1),
            })

    logger.info("Total prompt embed samples: {}", len(all_samples))

    # ------------------------------------------------------------------
    # Filter to samples that have VAE latents (inner join)
    # ------------------------------------------------------------------
    logger.info("Scanning VAE latent directory …")
    vae_available = set(
        f.name for f in vae_dir.iterdir() if f.suffix == ".safetensors"
    )
    logger.info("Found {} VAE latent files", len(vae_available))

    complete: list[dict] = []
    for s in all_samples:
        vae_filename = f"{s['tar_stem']}_{s['index_in_tar']}.safetensors"
        if vae_filename in vae_available:
            s["vae_path"] = str(vae_dir / vae_filename)
            complete.append(s)

    missing = len(all_samples) - len(complete)
    logger.info(
        "Complete samples (have both): {} / {} (missing VAE: {})",
        len(complete), len(all_samples), missing,
    )
    if not complete:
        logger.error("No complete samples found!")
        return

    # ------------------------------------------------------------------
    # Shuffle for training
    # ------------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(complete)

    # ------------------------------------------------------------------
    # Write tar shards
    # ------------------------------------------------------------------
    num_shards = (len(complete) + args.samples_per_shard - 1) // args.samples_per_shard
    logger.info("Writing {} shards (~{} samples each) …", num_shards, args.samples_per_shard)

    pe_handles: dict[str, safe_open] = {}

    def get_pe(path: str):
        if path not in pe_handles:
            pe_handles[path] = safe_open(path, framework="pt")
        return pe_handles[path]

    tar_file = None
    shard_id = -1
    total_bytes = 0

    for global_idx, sample in enumerate(tqdm(complete, desc="Writing shards")):
        # Start new shard if needed
        if global_idx % args.samples_per_shard == 0:
            if tar_file is not None:
                tar_file.close()
            shard_id += 1
            tar_file = tarfile.open(str(output_dir / f"shard-{shard_id:06d}.tar"), "w")

        # Read prompt embeds from cached handle
        prompt_embeds = get_pe(sample["source_file"]).get_tensor(sample["key"])

        # Read VAE latents
        with safe_open(sample["vae_path"], framework="pt") as vae_sf:
            latents = vae_sf.get_tensor("latents")
            condition = vae_sf.get_tensor("condition")

        # Serialize tensors
        st_bytes = st_save({
            "prompt_embeds": prompt_embeds,
            "latents": latents,
            "condition": condition,
        })

        # Serialize metadata
        meta_bytes = json.dumps({
            "prompt": sample["prompt"],
            "tar": sample["tar"],
            "index_in_tar": sample["index_in_tar"],
            "seq_len": sample["seq_len"],
        }).encode()

        key = f"{global_idx:07d}"
        _add_tar_entry(tar_file, f"{key}.safetensors", st_bytes)
        _add_tar_entry(tar_file, f"{key}.json", meta_bytes)
        total_bytes += len(st_bytes) + len(meta_bytes)

    if tar_file is not None:
        tar_file.close()
    pe_handles.clear()

    logger.info(
        "Done! {} shards, {} samples, {:.1f} GB → {}",
        shard_id + 1, len(complete), total_bytes / 1e9, output_dir,
    )


if __name__ == "__main__":
    main()
