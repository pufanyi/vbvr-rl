"""Judges score individual EvalSample objects."""

from .base import Judge
from .vlm import VLMJudge

__all__ = ["Judge", "VLMJudge"]
