"""Uniform frame sampling from MP4 files."""

from __future__ import annotations

from pathlib import Path

import decord
import numpy as np
from PIL import Image


def sample_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    """
    Uniformly sample `num_frames` frames from a video (clamped to total frame count).
    Returns PIL RGB images ready for VLM processors.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")

    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total == 0:
        return []

    k = min(num_frames, total)
    indices = np.linspace(0, total - 1, k, dtype=int).tolist()
    batch = vr.get_batch(indices).asnumpy()  # (k, H, W, 3) uint8 RGB
    return [Image.fromarray(batch[i]) for i in range(batch.shape[0])]
