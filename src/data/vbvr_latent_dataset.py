"""Dataset for VBVR precomputed latents in WebDataset tar-shard format.

Each tar shard contains pairs of files per sample:
  {key}.safetensors  — prompt_embeds, latents, condition
                     or prompt_embeds, latents_0, latents_1, ..., condition
  {key}.json         — prompt, tar, index_in_tar, seq_len

Usage:
    dataset = VBVRLatentDataset("data/vbvr/latents/webdataset")
"""

import logging
import re
from pathlib import Path

import torch.nn.functional as F
import webdataset as wds
from safetensors.torch import load as st_load
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)
_LATENTS_KEY_RE = re.compile(r"latents_(\d+)$")


_RESERVED_TENSOR_KEYS = frozenset({"prompt_embeds", "condition", "latents"})


def _decode_sample(sample: dict, max_text_len: int = 512) -> dict:
    """Decode a single webdataset sample into training tensors.

    Any non-reserved tensor in the safetensors blob (e.g. ``maze_*`` fields
    emitted by ``src.precompute.maze_webdataset``) is passed through verbatim
    so downstream rewards can consume it.  Callers that don't care simply
    ignore the extra keys.
    """
    tensors = st_load(sample["safetensors"])
    prompt_embeds = tensors["prompt_embeds"]

    seq_len = prompt_embeds.shape[0]
    if seq_len < max_text_len:
        prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, max_text_len - seq_len))
    elif seq_len > max_text_len:
        prompt_embeds = prompt_embeds[:max_text_len]

    decoded = {
        "prompt_embeds": prompt_embeds,
        "condition": tensors["condition"],
    }

    if "latents" in tensors:
        decoded["video_latents"] = tensors["latents"]
    else:
        latent_keys = []
        for key in tensors:
            match = _LATENTS_KEY_RE.fullmatch(key)
            if match is not None:
                latent_keys.append((int(match.group(1)), key))
        if not latent_keys:
            raise KeyError("Expected either 'latents' or 'latents_<n>' tensors in latent sample")
        latent_keys.sort()
        decoded["video_latents"] = [tensors[key] for _, key in latent_keys]

    # Pass through anything else (e.g. maze_* reward metadata).
    for key, value in tensors.items():
        if key in _RESERVED_TENSOR_KEYS or _LATENTS_KEY_RE.fullmatch(key):
            continue
        decoded[key] = value
    return decoded


class VBVRLatentDataset(IterableDataset):
    """WebDataset-based loader for precomputed VAE latents + T5 prompt embeddings.

    Composes a ``wds.WebDataset`` pipeline with shard discovery, distributed
    splitting, shuffle, and decoding.  This is an IterableDataset — set
    ``dataset_size`` in TrainConfig for LR scheduling.

    Args:
        webdataset_dir: Directory containing ``shard-NNNNNN.tar`` files.
        max_text_len: Pad/truncate prompt embeddings to this length.
        shuffle_buffer: Buffer size for sample-level shuffle (0 to disable).
        epoch_length: If set, cap each rank to exactly this many samples per
            epoch via ``with_epoch()``.  This guarantees all ranks produce the
            same number of batches, preventing FSDP/NCCL deadlocks at epoch
            boundaries when shard counts are not evenly divisible by world_size.
    """

    def __init__(
        self,
        webdataset_dir: str,
        max_text_len: int = 512,
        shuffle_buffer: int = 50000,
        epoch_length: int | None = None,
    ):
        shard_dir = Path(webdataset_dir)
        shard_paths = sorted(shard_dir.glob("shard-*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard-*.tar files found in {shard_dir}")

        urls = [str(p) for p in shard_paths]
        logger.info("VBVRLatentDataset: %d shards in %s", len(shard_paths), shard_dir)

        pipeline = wds.WebDataset(urls, nodesplitter=wds.split_by_node, shardshuffle=len(urls))
        if shuffle_buffer > 0:
            pipeline = pipeline.shuffle(shuffle_buffer)
        pipeline = pipeline.map(lambda s: _decode_sample(s, max_text_len))
        if epoch_length is not None:
            pipeline = pipeline.with_epoch(epoch_length)
            logger.info("VBVRLatentDataset: epoch_length=%d samples per rank", epoch_length)
        self._pipeline = pipeline

    def __iter__(self):
        return iter(self._pipeline)
