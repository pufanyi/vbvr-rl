"""Data types for VBVR-Bench evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["In_Domain", "Out_of_Domain"]
Split = Literal["Open_60", "Hidden_40"]


class EvalSample(BaseModel):
    """One (task × video) pair to score."""

    task_name: str
    video_idx: str
    split: Split
    domain: Domain
    video_path: Path
    gt_dir: Path
    gt_first_frame: Path
    gt_final_frame: Path
    gt_video_path: Path | None = None
    prompt: str


class SampleScore(BaseModel):
    """Score for a single EvalSample."""

    task_name: str
    video_idx: str
    split: Split
    domain: Domain
    score: float = Field(ge=0.0, le=1.0)
    judge_response: str | None = None
    details: dict = Field(default_factory=dict)
    error: str | None = None


class DomainSummary(BaseModel):
    """Aggregated stats for a domain (or overall)."""

    mean_score: float
    num_samples: int
    by_task: dict[str, float] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Full eval run result — matches VBVR-EvalKit's eval_results.json shape."""

    model_name: str
    judge: str
    timestamp: str
    samples: list[SampleScore]
    In_Domain: DomainSummary
    Out_of_Domain: DomainSummary
    overall: DomainSummary
