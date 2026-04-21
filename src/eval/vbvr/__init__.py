"""VBVR-Bench evaluation — VLM-judge based scoring for I2V models."""

from .dataset import discover_samples
from .runner import run_eval
from .types import EvalSample, RunSummary, SampleScore

__all__ = [
    "EvalSample",
    "RunSummary",
    "SampleScore",
    "discover_samples",
    "run_eval",
]
