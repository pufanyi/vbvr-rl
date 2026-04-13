"""Dataset for VBVR precomputed latents in WebDataset tar-shard format.

Each tar shard contains pairs of files per sample:
  {key}.safetensors  — prompt_embeds, latents, condition
  {key}.json         — prompt, tar, index_in_tar, seq_len

Usage:
    dataset = VBVRLatentDataset("data/vbvr/latents/webdataset")
"""

import logging
from pathlib import Path

import torch.nn.functional as F
import webdataset as wds
from safetensors.torch import load as st_load
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


def _decode_sample(sample: dict, max_text_len: int = 512) -> dict:
    """Decode a single webdataset sample into training tensors."""
    tensors = st_load(sample["safetensors"])
    prompt_embeds = tensors["prompt_embeds"]

    seq_len = prompt_embeds.shape[0]
    if seq_len < max_text_len:
        prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, max_text_len - seq_len))
    elif seq_len > max_text_len:
        prompt_embeds = prompt_embeds[:max_text_len]

    return {
        "prompt_embeds": prompt_embeds,
        "video_latents": tensors["latents"],
        "condition": tensors["condition"],
    }


class VBVRLatentDataset(IterableDataset):
    """WebDataset-based loader for precomputed VAE latents + T5 prompt embeddings.

    Composes a ``wds.WebDataset`` pipeline with shard discovery, distributed
    splitting, shuffle, and decoding.  This is an IterableDataset — set
    ``dataset_size`` in TrainConfig for LR scheduling.

    Args:
        webdataset_dir: Directory containing ``shard-NNNNNN.tar`` files.
        max_text_len: Pad/truncate prompt embeddings to this length.
        shuffle_buffer: Buffer size for sample-level shuffle (0 to disable).
    """

    def __init__(
        self,
        webdataset_dir: str,
        max_text_len: int = 512,
        shuffle_buffer: int = 1000,
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
        self._pipeline = pipeline.map(lambda s: _decode_sample(s, max_text_len))

    def __iter__(self):
        return iter(self._pipeline)
