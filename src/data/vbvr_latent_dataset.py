"""Dataset for VBVR precomputed latents in WebDataset tar-shard format.

Each tar shard contains pairs of files per sample:
  {key}.safetensors  — prompt_embeds, latents, condition
                     or prompt_embeds, latents_0, latents_1, ..., condition
  {key}.json         — prompt, tar, index_in_tar, seq_len

Usage:
    dataset = VBVRLatentDataset("data/vbvr/latents/webdataset")
"""

import json
import logging
import re
from itertools import islice
from pathlib import Path

import torch.nn.functional as F
import webdataset as wds
from safetensors.torch import load as st_load
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)
_LATENTS_KEY_RE = re.compile(r"latents_(\d+)$")


_RESERVED_TENSOR_KEYS = frozenset({"prompt_embeds", "condition", "latents"})


def _json_metadata(sample: dict) -> dict | None:
    metadata = sample.get("json")
    if isinstance(metadata, (bytes, bytearray)):
        try:
            metadata = json.loads(metadata.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    return metadata if isinstance(metadata, dict) else None


def _task_name_from_metadata(metadata: dict | None) -> str:
    if metadata is None:
        return ""
    if "task_name" in metadata:
        return str(metadata["task_name"])
    tar_name = metadata.get("tar", "")
    return Path(str(tar_name)).stem if tar_name else ""


def _task_allowed(sample: dict, allowed_task_names: frozenset[str] | None) -> bool:
    if allowed_task_names is None:
        return True
    task_name = _task_name_from_metadata(_json_metadata(sample))
    return task_name in allowed_task_names


class _RankShardSplitter:
    """Split WebDataset shards with an explicit rank/world-size pair."""

    def __init__(self, rank: int, world_size: int):
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __call__(self, src, group=None):
        if self.world_size > 1:
            yield from islice(src, self.rank, None, self.world_size)
        else:
            yield from src


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
    if "__key__" in sample:
        decoded["sample_key"] = sample["__key__"]
    if "__url__" in sample:
        decoded["sample_url"] = sample["__url__"]

    metadata = _json_metadata(sample)
    if isinstance(metadata, dict):
        for key in ("tar", "index_in_tar", "seq_len", "prompt"):
            if key in metadata:
                decoded[f"sample_{key}"] = metadata[key]

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
            epoch.  The cap is split across DataLoader workers so multi-worker
            loading does not multiply the effective epoch length.
        seed: Deterministic shard/sample shuffle seed.
        allowed_task_names: If set, drop samples whose JSON metadata task name
            is not in this allowlist before tensor decoding.
        node_rank: Explicit distributed rank for shard splitting.  Expert
            parallel duplicate mode passes the rank within an expert group so
            the high and low groups see the same shard stream.
        node_world_size: Explicit distributed world size for shard splitting.
    """

    def __init__(
        self,
        webdataset_dir: str,
        max_text_len: int = 512,
        shuffle_buffer: int = 50000,
        epoch_length: int | None = None,
        seed: int | None = None,
        allowed_task_names: set[str] | frozenset[str] | None = None,
        node_rank: int | None = None,
        node_world_size: int | None = None,
    ):
        shard_dir = Path(webdataset_dir)
        shard_paths = sorted(shard_dir.glob("shard-*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard-*.tar files found in {shard_dir}")

        urls = [str(p) for p in shard_paths]
        logger.info("VBVRLatentDataset: %d shards in %s", len(shard_paths), shard_dir)

        if (node_rank is None) != (node_world_size is None):
            raise ValueError("node_rank and node_world_size must be provided together")
        rank_shard_count = None
        if node_rank is None:
            nodesplitter = wds.split_by_node
            logger.info("VBVRLatentDataset: shard splitting by global PyTorch rank")
        else:
            nodesplitter = _RankShardSplitter(node_rank, node_world_size)
            rank_shard_count = (len(urls) + node_world_size - 1 - node_rank) // node_world_size
            logger.info(
                "VBVRLatentDataset: shard splitting rank=%d/%d (%d shards)",
                node_rank,
                node_world_size,
                rank_shard_count,
            )

        pipeline = wds.WebDataset(
            urls,
            nodesplitter=nodesplitter,
            shardshuffle=len(urls),
            seed=seed,
            empty_check=False,
        )
        allowed_tasks = frozenset(allowed_task_names) if allowed_task_names is not None else None
        if allowed_tasks is not None:
            logger.info("VBVRLatentDataset: filtering to %d allowed task names", len(allowed_tasks))
            pipeline = pipeline.select(lambda sample: _task_allowed(sample, allowed_tasks))
        if shuffle_buffer > 0:
            pipeline = pipeline.shuffle(shuffle_buffer, seed=seed)
        pipeline = pipeline.map(lambda s: _decode_sample(s, max_text_len))
        self._pipeline = pipeline
        self._epoch_length = epoch_length
        self._rank_shard_count = rank_shard_count
        if epoch_length is not None:
            logger.info("VBVRLatentDataset: epoch_length=%d samples per rank", epoch_length)

    def _worker_epoch_length(self) -> int | None:
        """Return this DataLoader worker's share of the per-rank epoch length."""
        if self._epoch_length is None:
            return None

        worker = get_worker_info()
        if worker is None or worker.num_workers <= 1:
            return self._epoch_length

        active_workers = worker.num_workers
        if self._rank_shard_count is not None:
            active_workers = min(active_workers, self._rank_shard_count)
        active_workers = min(active_workers, self._epoch_length)
        if active_workers <= 0 or worker.id >= active_workers:
            return 0

        base, extra = divmod(self._epoch_length, active_workers)
        return base + int(worker.id < extra)

    def __iter__(self):
        limit = self._worker_epoch_length()
        if limit is None:
            yield from self._pipeline
            return

        yielded = 0
        while yielded < limit:
            produced = 0
            for sample in self._pipeline:
                yield sample
                yielded += 1
                produced += 1
                if yielded >= limit:
                    break
            if produced == 0:
                break
