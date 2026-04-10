"""Dataset for precomputed latents stored in parquet.

Config JSON format (same structure as I2VDataset):
    [{"data_path": "/path/to/latents.parquet"}, ...]

Parquet schema (produced by scripts/precompute_latents.py):
    - prompt               (string)
    - prompt_embeds        (bytes) — bf16 tensor
    - video_latents_0      (bytes) — bf16 tensor, step 0
    - video_latents_1      (bytes) — bf16 tensor, step 1
    - ...
    - final_latents        (bytes) — bf16 tensor, final step
    - condition            (bytes) — bf16 tensor
    - num_steps            (int)
    - embed_shape          (string, JSON list)
    - latent_shape         (string, JSON list)
    - condition_shape      (string, JSON list)
"""

import bisect
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _bytes_to_bf16(data: bytes, shape: list[int]) -> torch.Tensor:
    """Deserialize bytes to a bf16 tensor with the given shape."""
    arr = np.frombuffer(data, dtype=np.uint16).copy()
    return torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)


class LatentDataset(Dataset):
    """Parquet-backed dataset of precomputed latents. Zero decoding overhead."""

    def __init__(self, json_path: str):
        config_path = Path(json_path)
        raw = json.loads(config_path.read_text())
        entries = [raw] if isinstance(raw, dict) else raw

        self._tables = []
        self._cumulative: list[int] = []
        total = 0

        for entry in entries:
            data_path = Path(entry["data_path"])
            if not data_path.is_absolute():
                data_path = config_path.parent / data_path

            table = pq.read_table(data_path)
            n = table.num_rows
            self._tables.append(table)
            total += n
            self._cumulative.append(total)
            logger.info("Loaded %d precomputed rows from %s", n, data_path)

        self._len = total

    def _locate(self, idx: int) -> tuple[int, int]:
        ti = bisect.bisect_right(self._cumulative, idx)
        local = idx if ti == 0 else idx - self._cumulative[ti - 1]
        return ti, local

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        ti, row = self._locate(idx)
        table = self._tables[ti]

        prompt = table.column("prompt")[row].as_py()
        num_steps = table.column("num_steps")[row].as_py()

        embed_shape = json.loads(table.column("embed_shape")[row].as_py())
        latent_shape = json.loads(table.column("latent_shape")[row].as_py())
        condition_shape = json.loads(table.column("condition_shape")[row].as_py())

        prompt_embeds = _bytes_to_bf16(table.column("prompt_embeds")[row].as_py(), embed_shape)
        condition = _bytes_to_bf16(table.column("condition")[row].as_py(), condition_shape)

        video_latents = []
        for step_idx in range(num_steps - 1):
            vl = _bytes_to_bf16(table.column(f"video_latents_{step_idx}")[row].as_py(), latent_shape)
            video_latents.append(vl)
        video_latents.append(_bytes_to_bf16(table.column("final_latents")[row].as_py(), latent_shape))

        return {
            "index": idx,
            "prompt": prompt,
            "prompt_embeds": prompt_embeds,
            "video_latents": video_latents,
            "condition": condition,
        }
