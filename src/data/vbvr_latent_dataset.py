"""Dataset for VBVR precomputed latents (separate VAE + T5 safetensors).

Config JSON format:
    {
        "vae_latents_dir": "/path/to/vae_latents",
        "prompt_embeds_dir": "/path/to/prompt_embeds",
        "max_text_len": 512
    }

VAE latents dir contains per-sample files: {tar_stem}_{idx}.safetensors
    - latents   (bf16)
    - condition (bf16)
    - metadata: prompt, tar, index_in_tar

Prompt embeds dir contains per-batch files: rank{R}_batch{B}.safetensors
    - "0", "1", ... (variable-length bf16 embeddings)
    - metadata: count, samples (JSON list with tar, index_in_tar, prompt)
"""

import json
import logging
from pathlib import Path

import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VBVRLatentDataset(Dataset):
    """Loads precomputed VAE latents and T5 prompt embeddings from VBVR format."""

    def __init__(self, json_path: str):
        config_path = Path(json_path)
        raw = json.loads(config_path.read_text())
        if isinstance(raw, list):
            raw = raw[0]

        self._max_text_len = raw.get("max_text_len", 512)

        vae_dir = Path(raw["vae_latents_dir"])
        embeds_dir = Path(raw["prompt_embeds_dir"])

        if not vae_dir.is_absolute():
            vae_dir = config_path.parent / vae_dir
        if not embeds_dir.is_absolute():
            embeds_dir = config_path.parent / embeds_dir

        # ---- Index VAE latent files ----
        vae_files = sorted(vae_dir.glob("*.safetensors"))
        self._vae_files = vae_files
        logger.info("VBVRLatentDataset: %d VAE latent files from %s", len(vae_files), vae_dir)

        # ---- Build prompt embeds index: (tar_stem, index_in_tar) -> (file, key) ----
        self._embed_index: dict[tuple[str, int], tuple[Path, str]] = {}
        embed_files = sorted(embeds_dir.glob("*.safetensors"))
        for ef in embed_files:
            f = safe_open(str(ef), framework="pt")
            metadata = f.metadata()
            if metadata is None:
                continue
            samples = json.loads(metadata.get("samples", "[]"))
            for i, s in enumerate(samples):
                tar_stem = Path(s["tar"]).stem
                idx_in_tar = s["index_in_tar"]
                self._embed_index[(tar_stem, idx_in_tar)] = (ef, str(i))
            f = None  # close

        logger.info(
            "VBVRLatentDataset: %d prompt embed entries from %d files in %s",
            len(self._embed_index),
            len(embed_files),
            embeds_dir,
        )

    def __len__(self):
        return len(self._vae_files)

    def __getitem__(self, idx):
        vae_path = self._vae_files[idx]

        # Load VAE latents
        with safe_open(str(vae_path), framework="pt") as f:
            latents = f.get_tensor("latents")
            condition = f.get_tensor("condition")
            metadata = f.metadata()

        tar_name = metadata.get("tar", "")
        index_in_tar = int(metadata.get("index_in_tar", "0"))
        tar_stem = Path(tar_name).stem

        # Load prompt embeddings
        embed_key = (tar_stem, index_in_tar)
        if embed_key in self._embed_index:
            embed_file, tensor_key = self._embed_index[embed_key]
            with safe_open(str(embed_file), framework="pt") as f:
                prompt_embeds = f.get_tensor(tensor_key)
        else:
            raise KeyError(
                f"No prompt embedding found for ({tar_stem}, {index_in_tar}). "
                f"Ensure prompt_embeds_dir covers all samples in vae_latents_dir."
            )

        # Pad or truncate to fixed length so batches can be stacked
        seq_len, dim = prompt_embeds.shape
        if seq_len < self._max_text_len:
            prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, self._max_text_len - seq_len))
        elif seq_len > self._max_text_len:
            prompt_embeds = prompt_embeds[: self._max_text_len]

        return {
            "prompt_embeds": prompt_embeds,
            "video_latents": latents,
            "condition": condition,
        }
