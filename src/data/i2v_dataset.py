"""I2V training dataset — parquet-native.

Config JSON points to one or more parquet files:

    Single dataset:
    {
        "data_path": "/path/to/train.parquet",
        "root": "/path/to/video/root"        // base dir for relative paths
    }

    Multi-dataset (list):
    [
        {"data_path": "/path/to/a/train.parquet", "root": "/path/to/a/"},
        {"data_path": "/path/to/b/train.parquet", "root": "/path/to/b/"}
    ]

Parquet schema:
    - videos: list<string>  — [search_video, video] (2 entries) or [video] (1 entry)
      OR video: string      — single video path
    - prompt: string
    - image:  string        — optional, reference image path (uses first video frame if absent)

Per-dataset overrides (optional keys in the config dict):
    num_frames, max_area, height, width, fps
"""

import bisect
import json
import logging
import random
from pathlib import Path

import decord
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

decord.bridge.set_bridge("torch")

# Height/width must be divisible by vae_scale_factor_spatial * patch_size.
# For Wan2.2: 8 * 2 = 16.
_MOD_VALUE = 16


class _ItemConfig(BaseModel):
    num_frames: int
    max_area: int
    fixed_height: int | None = None
    fixed_width: int | None = None
    fps: int


def compute_hw(max_area: int, aspect_ratio: float) -> tuple[int, int]:
    """Compute (height, width) from a pixel budget and aspect ratio (h/w)."""
    height = round(np.sqrt(max_area * aspect_ratio)) // _MOD_VALUE * _MOD_VALUE
    width = round(np.sqrt(max_area / aspect_ratio)) // _MOD_VALUE * _MOD_VALUE
    height = max(height, _MOD_VALUE)
    width = max(width, _MOD_VALUE)
    return height, width


class I2VDataset(Dataset):
    """Parquet-backed video dataset. Rows are read directly from Arrow tables."""

    def __init__(
        self,
        json_path: str,
        num_frames: int | None = None,
        max_area: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
    ):
        config_path = Path(json_path)
        parent_dir = config_path.parent
        raw = json.loads(config_path.read_text())

        if isinstance(raw, dict):
            entries = [raw]
        elif isinstance(raw, list):
            entries = raw
        else:
            raise ValueError(f"Config JSON must be a dict or list of dicts: {config_path}")

        # Per-table metadata
        self._tables: list[pq.ParquetFile] = []
        self._roots: list[Path] = []
        self._configs: list[_ItemConfig] = []
        self._cumulative: list[int] = []  # cumulative row counts

        total = 0
        for entry in entries:
            data_path = Path(entry["data_path"])
            if not data_path.is_absolute():
                data_path = parent_dir / data_path

            table = pq.read_table(data_path)
            n = table.num_rows

            # Resolve root directory
            if "root" in entry:
                root = Path(entry["root"])
                if not root.is_absolute():
                    root = parent_dir / root
            else:
                root = data_path.parent

            cfg = _ItemConfig(
                num_frames=num_frames if num_frames is not None else entry.get("num_frames", 81),
                max_area=max_area if max_area is not None else entry.get("max_area", 480 * 832),
                fixed_height=height if height is not None else entry.get("height"),
                fixed_width=width if width is not None else entry.get("width"),
                fps=fps if fps is not None else entry.get("fps", 16),
            )

            self._tables.append(table)
            self._roots.append(root)
            self._configs.append(cfg)
            total += n
            self._cumulative.append(total)

            logger.info("Loaded %d rows from %s (root=%s)", n, data_path, root)

        self._len = total

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map global index → (table_index, local_row)."""
        ti = bisect.bisect_right(self._cumulative, idx)
        local = idx if ti == 0 else idx - self._cumulative[ti - 1]
        return ti, local

    # ------------------------------------------------------------------
    # Row reading
    # ------------------------------------------------------------------

    def _read_row(self, ti: int, row: int) -> dict:
        """Read a single row from the Arrow table as a Python dict."""
        table = self._tables[ti]
        cols = table.column_names
        result: dict = {}

        if "videos" in cols:
            videos = table.column("videos")[row].as_py()
            if len(videos) >= 2:
                result["search_video"] = videos[0]
                result["video"] = videos[1]
            else:
                result["video"] = videos[0]
        elif "video" in cols:
            result["video"] = table.column("video")[row].as_py()
        else:
            raise ValueError(f"Table {ti} has no 'videos' or 'video' column")

        result["prompt"] = table.column("prompt")[row].as_py() if "prompt" in cols else ""

        if "image" in cols:
            val = table.column("image")[row].as_py()
            if val is not None:
                result["image"] = val

        return result

    # ------------------------------------------------------------------
    # Path / media helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(path: str, root: Path) -> str:
        p = Path(path)
        return str(p) if p.is_absolute() else str(root / p)

    @staticmethod
    def _get_video_hw(video_path: str, cfg: _ItemConfig) -> tuple[int, int]:
        if cfg.fixed_height is not None and cfg.fixed_width is not None:
            return cfg.fixed_height, cfg.fixed_width
        vr = decord.VideoReader(video_path)
        orig_h, orig_w = vr[0].shape[:2]
        return compute_hw(cfg.max_area, orig_h / orig_w)

    @staticmethod
    def _load_video(video_path: str, height: int, width: int, cfg: _ItemConfig) -> torch.Tensor:
        """Load video frames as uint8. Returns (C, T, H, W)."""
        vr = decord.VideoReader(video_path, width=width, height=height)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, cfg.num_frames).round().astype(int).tolist()
        frames = vr.get_batch(indices)  # (T, H, W, C)
        return frames.permute(3, 0, 1, 2).contiguous()

    @staticmethod
    def _load_image(path: str, height: int, width: int) -> torch.Tensor:
        """Load a single image as uint8. Returns (C, H, W)."""
        with Image.open(path) as img:
            img = img.convert("RGB").resize((width, height), Image.LANCZOS)
            array = np.array(img, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self._len

    _MAX_RETRIES = 10

    def __getitem__(self, idx):
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._load_item(idx)
            except Exception:
                logger.warning(
                    "Failed to load item %d (attempt %d/%d), trying another sample.",
                    idx,
                    attempt + 1,
                    self._MAX_RETRIES,
                    exc_info=True,
                )
                idx = random.randint(0, self._len - 1)
        return self._load_item(idx)

    def _load_item(self, idx):
        ti, row = self._locate(idx)
        item = self._read_row(ti, row)
        cfg = self._configs[ti]
        root = self._roots[ti]

        video_path = self._resolve(item["video"], root)
        height, width = self._get_video_hw(video_path, cfg)
        video = self._load_video(video_path, height, width, cfg)

        raw_image = item.get("image")
        image = (
            self._load_image(self._resolve(raw_image, root), height, width) if raw_image else video[:, 0].clone()
        )

        result = {
            "index": idx,
            "video": video,
            "image": image,
            "prompt": item["prompt"],
        }

        if "search_video" in item:
            search_path = self._resolve(item["search_video"], root)
            result["search_video"] = self._load_video(search_path, height, width, cfg)

        return result
