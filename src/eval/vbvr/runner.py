"""Driver that scores all samples and writes VBVR-compatible output JSONs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .dataset import discover_samples
from .judges.base import Judge
from .types import DomainSummary, RunSummary, SampleScore


def _aggregate(scores: list[SampleScore], domain_filter: str | None) -> DomainSummary:
    selected = [s for s in scores if domain_filter is None or s.domain == domain_filter]
    by_task: dict[str, list[float]] = defaultdict(list)
    for s in selected:
        by_task[s.task_name].append(s.score)
    mean_by_task = {t: sum(v) / len(v) for t, v in by_task.items()}
    mean = sum(s.score for s in selected) / len(selected) if selected else 0.0
    return DomainSummary(mean_score=mean, num_samples=len(selected), by_task=mean_by_task)


def run_eval(
    model_output: Path,
    gt_base: Path,
    judge: Judge,
    output_dir: Path,
    tasks: list[str] | None = None,
    limit: int | None = None,
) -> RunSummary:
    """
    Score every (task, video) pair under `model_output`, write results and summary.

    Output layout (mirrors VBVR-EvalKit):
      <output_dir>/<model_name>/eval_results.json  — per-sample + summary
      <output_dir>/<model_name>/summary.json       — headline aggregates only
    """
    model_name = model_output.name
    samples = discover_samples(model_output, gt_base, tasks=tasks)
    if limit is not None:
        samples = samples[:limit]
        logger.info("Limiting to first {} samples", len(samples))

    scored: list[SampleScore] = []
    for sample in tqdm(samples, desc=f"[{model_name}|{judge.name}]"):
        scored.append(judge.score(sample))

    summary = RunSummary(
        model_name=model_name,
        judge=judge.name,
        timestamp=datetime.now().isoformat(),
        samples=scored,
        In_Domain=_aggregate(scored, "In_Domain"),
        Out_of_Domain=_aggregate(scored, "Out_of_Domain"),
        overall=_aggregate(scored, None),
    )

    out_dir = output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(summary.model_dump_json(indent=2))
    headline = {
        "model_name": summary.model_name,
        "judge": summary.judge,
        "timestamp": summary.timestamp,
        "In_Domain": summary.In_Domain.model_dump(),
        "Out_of_Domain": summary.Out_of_Domain.model_dump(),
        "overall": summary.overall.model_dump(),
    }
    (out_dir / "summary.json").write_text(json.dumps(headline, indent=2))

    logger.info(
        "Done. In_Domain={:.4f} ({}) | Out_of_Domain={:.4f} ({}) | overall={:.4f} ({})",
        summary.In_Domain.mean_score,
        summary.In_Domain.num_samples,
        summary.Out_of_Domain.mean_score,
        summary.Out_of_Domain.num_samples,
        summary.overall.mean_score,
        summary.overall.num_samples,
    )
    return summary
