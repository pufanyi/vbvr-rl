"""I2V training dataset — parquet-native.

Config JSON points to one or more parquet files:

    Single dataset:
    {"data_path": "/path/to/train.parquet", "root": "/path/to/video/root"}

    Multi-dataset (list):
    [
        {"data_path": "/path/to/a/train.parquet", "root": "/path/to/a/"},
        {"data_path": "/path/to/b/train.parquet", "root": "/path/to/b/"}
    ]

Parquet schema:
    - videos: list<string>  — ordered video paths [step_0, step_1, ..., final]
      OR video: string      — single video path (equivalent to [video])
    - prompt: string
    - image:  string        — optional reference image (uses first frame of videos[-1] if absent)

Per-dataset overrides (optional keys in the config dict):
    num_frames, max_area, height, width, fps
"""

import bisect
import faulthandler
import json
import logging
import os
import random
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import Dataset

from src.data.remote_io import is_remote_path, localize_media_path, resolve_media_path

logger = logging.getLogger(__name__)

try:
    import decord  # type: ignore
except Exception:
    decord = None

_HAS_DECORD = decord is not None and hasattr(decord, "VideoReader")
if _HAS_DECORD and hasattr(decord, "bridge") and hasattr(decord.bridge, "set_bridge"):
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


def _decord_num_threads() -> int:
    value = os.environ.get("WAN_TRAINER_DECORD_NUM_THREADS", "1")
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning("Invalid WAN_TRAINER_DECORD_NUM_THREADS=%r; using 1", value)
        return 1


class _SlowItemWatchdog:
    def __init__(self, seconds: int, message: str):
        self._seconds = seconds
        self._message = message
        self._timer: threading.Timer | None = None

    def start(self) -> "_SlowItemWatchdog":
        if self._seconds <= 0:
            return self
        self._timer = threading.Timer(self._seconds, self._dump)
        self._timer.daemon = True
        self._timer.start()
        return self

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _dump(self) -> None:
        print(self._message, file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


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
        shuffle_indices: bool = False,
        shuffle_seed: int = 42,
        remote_prefetch_lookahead: int = 0,
        remote_prefetch_workers: int = 1,
        remote_prefetch_stride: int = 1,
        item_trace_seconds: int = 0,
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

        self._tables: list[pq.ParquetFile] = []
        self._roots: list[Path] = []
        self._configs: list[_ItemConfig] = []
        self._cumulative: list[int] = []

        total = 0
        for entry in entries:
            data_path = Path(entry["data_path"])
            if not data_path.is_absolute():
                data_path = parent_dir / data_path

            table = pq.read_table(data_path)
            n = table.num_rows

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
        self._index_map: np.ndarray | None = None
        self._remote_prefetch_lookahead = max(0, int(remote_prefetch_lookahead))
        self._remote_prefetch_workers = max(1, int(remote_prefetch_workers))
        self._remote_prefetch_stride = max(1, int(remote_prefetch_stride))
        self._item_trace_seconds = max(0, int(item_trace_seconds))
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_futures: dict[int, Future] = {}
        if shuffle_indices:
            self._index_map = np.arange(total, dtype=np.int64)
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(self._index_map)
            logger.info("Shuffled %d raw dataset indices with seed=%d", total, shuffle_seed)

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map global index -> (table_index, local_row)."""
        ti = bisect.bisect_right(self._cumulative, idx)
        local = idx if ti == 0 else idx - self._cumulative[ti - 1]
        return ti, local

    def _source_index(self, idx: int) -> int:
        """Map logical shuffled index -> physical source row index."""
        idx = int(idx)
        if self._index_map is None:
            return idx
        return int(self._index_map[idx])

    # ------------------------------------------------------------------
    # Row reading
    # ------------------------------------------------------------------

    @staticmethod
    def _read_row(table, row: int) -> tuple[list[str], str, str | None]:
        """Read a single row. Returns (video_paths, prompt, image_path)."""
        cols = table.column_names

        if "videos" in cols:
            video_paths = table.column("videos")[row].as_py()
        elif "video" in cols:
            video_paths = [table.column("video")[row].as_py()]
        else:
            raise ValueError("Table has no 'videos' or 'video' column")

        prompt = table.column("prompt")[row].as_py() if "prompt" in cols else ""
        image = table.column("image")[row].as_py() if "image" in cols else None

        return video_paths, prompt, image

    # ------------------------------------------------------------------
    # Path / media helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(path: str, root: Path) -> str:
        return resolve_media_path(path, root)

    @staticmethod
    def _get_video_hw(video_path: str, cfg: _ItemConfig) -> tuple[int, int]:
        if cfg.fixed_height is not None and cfg.fixed_width is not None:
            return cfg.fixed_height, cfg.fixed_width
        video_path = localize_media_path(video_path)
        if _HAS_DECORD:
            vr = decord.VideoReader(video_path, num_threads=_decord_num_threads())
            orig_h, orig_w = vr[0].shape[:2]
        else:
            meta = iio.immeta(video_path)
            if "source_size" in meta:
                orig_w, orig_h = meta["source_size"]
            elif "size" in meta:
                orig_w, orig_h = meta["size"]
            else:
                _, orig_h, orig_w, _ = iio.improps(video_path).shape
        return compute_hw(cfg.max_area, orig_h / orig_w)

    @staticmethod
    def _load_video(video_path: str, height: int, width: int, cfg: _ItemConfig) -> torch.Tensor:
        """Load video frames as uint8. Returns (C, T, H, W)."""
        video_path = localize_media_path(video_path)
        if _HAS_DECORD:
            vr = decord.VideoReader(video_path, width=width, height=height, num_threads=_decord_num_threads())
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, cfg.num_frames).round().astype(int).tolist()
            frames = vr.get_batch(indices)  # (T, H, W, C)
            return frames.permute(3, 0, 1, 2).contiguous()

        frames = iio.imread(video_path)
        total_frames = int(frames.shape[0])
        indices = np.linspace(0, total_frames - 1, cfg.num_frames).round().astype(int)
        frames = frames[indices]
        if frames.shape[1] != height or frames.shape[2] != width:
            tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
            frames = tensor.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()

    @staticmethod
    def _load_image(path: str, height: int, width: int) -> torch.Tensor:
        """Load a single image as uint8. Returns (C, H, W)."""
        path = localize_media_path(path)
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
        self._schedule_remote_prefetch(int(idx))
        for attempt in range(self._MAX_RETRIES):
            watchdog = self._start_item_watchdog(idx, attempt + 1)
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
            finally:
                watchdog.cancel()
        watchdog = self._start_item_watchdog(idx, self._MAX_RETRIES + 1)
        try:
            return self._load_item(idx)
        finally:
            watchdog.cancel()

    def _start_item_watchdog(self, idx: int, attempt: int) -> _SlowItemWatchdog:
        if self._item_trace_seconds <= 0:
            return _SlowItemWatchdog(0, "")
        return _SlowItemWatchdog(
            self._item_trace_seconds,
            self._format_item_watchdog_message(idx, attempt),
        ).start()

    def _format_item_watchdog_message(self, idx: int, attempt: int) -> str:
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        try:
            source_idx = self._source_index(idx)
            ti, row = self._locate(source_idx)
            video_paths, prompt, image_path = self._read_row(self._tables[ti], row)
            root = self._roots[ti]
            paths = [self._resolve(p, root) for p in video_paths]
            if image_path is not None:
                paths.append(self._resolve(image_path, root))
            if len(paths) > 4:
                paths = paths[:4] + [f"... +{len(paths) - 4} more"]
            prompt = prompt.replace("\n", " ")[:160]
            detail = (
                f"logical_idx={idx} source_idx={source_idx} table={ti} row={row} "
                f"attempt={attempt}/{self._MAX_RETRIES + 1} paths={paths!r} prompt={prompt!r}"
            )
        except Exception as exc:  # noqa: BLE001
            detail = f"logical_idx={idx} attempt={attempt}/{self._MAX_RETRIES + 1} context_error={exc!r}"
        return (
            f"[I2VDataset watchdog] rank={rank} local_rank={local_rank} pid={os.getpid()} "
            f"item load exceeded {self._item_trace_seconds}s: {detail}"
        )

    def _ensure_prefetch_executor(self) -> ThreadPoolExecutor:
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=self._remote_prefetch_workers,
                thread_name_prefix="i2v-remote-prefetch",
            )
        return self._prefetch_executor

    def _prune_prefetch_futures(self) -> None:
        if not self._prefetch_futures:
            return
        done = [idx for idx, future in self._prefetch_futures.items() if future.done()]
        for idx in done:
            future = self._prefetch_futures.pop(idx)
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Remote prefetch failed for logical index %d: %r", idx, exc)

    def _schedule_remote_prefetch(self, idx: int) -> None:
        if self._remote_prefetch_lookahead <= 0:
            return
        self._prune_prefetch_futures()
        max_pending = self._remote_prefetch_lookahead * self._remote_prefetch_workers
        if len(self._prefetch_futures) >= max_pending:
            return

        executor = self._ensure_prefetch_executor()
        for offset in range(1, self._remote_prefetch_lookahead + 1):
            next_idx = idx + offset * self._remote_prefetch_stride
            if next_idx >= self._len:
                break
            if next_idx in self._prefetch_futures:
                continue
            self._prefetch_futures[next_idx] = executor.submit(self._prefetch_remote_media_for_index, next_idx)
            if len(self._prefetch_futures) >= max_pending:
                break

    def _prefetch_remote_media_for_index(self, idx: int) -> None:
        source_idx = self._source_index(idx)
        ti, row = self._locate(source_idx)
        video_paths, _prompt, image_path = self._read_row(self._tables[ti], row)
        root = self._roots[ti]
        paths = [self._resolve(p, root) for p in video_paths]
        if image_path is not None:
            paths.append(self._resolve(image_path, root))
        for path in paths:
            if is_remote_path(path):
                localize_media_path(path)

    def _load_item(self, idx):
        source_idx = self._source_index(idx)
        ti, row = self._locate(source_idx)
        video_paths, prompt, image_path = self._read_row(self._tables[ti], row)
        cfg = self._configs[ti]
        root = self._roots[ti]

        # Use the last video (final target) to determine resolution
        final_video_path = self._resolve(video_paths[-1], root)
        height, width = self._get_video_hw(final_video_path, cfg)

        # Load all videos in order
        videos = [self._load_video(self._resolve(p, root), height, width, cfg) for p in video_paths]

        # Reference image: explicit column, or first frame of the final video
        if image_path is not None:
            image = self._load_image(self._resolve(image_path, root), height, width)
        else:
            image = videos[-1][:, 0].clone()

        return {
            "index": source_idx,
            "videos": videos,
            "image": image,
            "prompt": prompt,
        }
