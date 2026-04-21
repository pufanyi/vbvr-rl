"""Driver that scores all samples and writes VBVR-compatible output JSONs.

Resumable by design: every score is appended to ``scores.jsonl`` as soon as
it lands, so a killed run picks up from where it left off on the next
invocation. The progress bar covers the full dataset (not just the unfinished
slice) via ``tqdm(initial=...)``.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .dataset import discover_samples
from .judges.base import Judge
from .types import DomainSummary, EvalSample, RunSummary, SampleScore


def _load_existing_scores(path: Path) -> list[SampleScore]:
    """Read JSONL, skipping blank or malformed lines (e.g. torn final line)."""
    if not path.exists():
        return []
    out: list[SampleScore] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(SampleScore.model_validate_json(line))
        except Exception as e:
            logger.warning("skip malformed line in {}: {}", path.name, e)
    return out


def _filter_remaining(
    samples: list[EvalSample],
    existing: list[SampleScore],
    retry_errors: bool,
) -> list[EvalSample]:
    """Return samples whose (task_name, video_idx) aren't already scored."""
    done: set[tuple[str, str]] = {
        (s.task_name, s.video_idx) for s in existing if not (retry_errors and s.error is not None)
    }
    return [s for s in samples if (s.task_name, s.video_idx) not in done]


def _aggregate(scores: list[SampleScore], domain_filter: str | None) -> DomainSummary:
    selected = [s for s in scores if domain_filter is None or s.domain == domain_filter]
    by_task: dict[str, list[float]] = defaultdict(list)
    for s in selected:
        by_task[s.task_name].append(s.score)
    mean_by_task = {t: sum(v) / len(v) for t, v in by_task.items()}
    mean = sum(s.score for s in selected) / len(selected) if selected else 0.0
    return DomainSummary(mean_score=mean, num_samples=len(selected), by_task=mean_by_task)


def _write_summary_files(out_dir: Path, summary: RunSummary) -> None:
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


def run_eval(
    model_output: Path,
    gt_base: Path,
    judge: Judge,
    output_dir: Path,
    tasks: list[str] | None = None,
    limit: int | None = None,
    fresh: bool = False,
    retry_errors: bool = False,
) -> RunSummary:
    """Score every (task, video) pair; resumable via ``scores.jsonl``.

    Output layout:
      <output_dir>/<model_name>/scores.jsonl     — append-only, one SampleScore per line
      <output_dir>/<model_name>/eval_results.json — derived from scores.jsonl at the end
      <output_dir>/<model_name>/summary.json     — headline aggregates

    Args:
        fresh: if True, back up the existing ``scores.jsonl`` to
            ``scores.jsonl.bak`` and start over.
        retry_errors: if True, re-score samples whose cached entry has an error.
    """
    model_name = model_output.name
    out_dir = output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"

    if fresh and scores_path.exists():
        backup = scores_path.with_suffix(".jsonl.bak")
        shutil.move(str(scores_path), str(backup))
        logger.info("fresh run — moved existing scores to {}", backup.name)

    samples = discover_samples(model_output, gt_base, tasks=tasks)
    if limit is not None:
        samples = samples[:limit]
        logger.info("Limiting to first {} samples", len(samples))

    existing = _load_existing_scores(scores_path)
    remaining = _filter_remaining(samples, existing, retry_errors=retry_errors)
    logger.info(
        "Resume state: {} already scored, {} to go (retry_errors={})",
        len(samples) - len(remaining),
        len(remaining),
        retry_errors,
    )

    desc = f"[{model_name}|{judge.name}]"
    with scores_path.open("a", encoding="utf-8") as f:
        pbar = tqdm(
            total=len(samples),
            initial=len(samples) - len(remaining),
            desc=desc,
            unit="sample",
        )
        try:
            for sample in remaining:
                score = judge.score(sample)
                f.write(score.model_dump_json() + "\n")
                f.flush()
                pbar.update(1)
        finally:
            pbar.close()

    # Rebuild aggregates from the authoritative JSONL (covers both pre-existing
    # and freshly-scored entries; order is append order).
    final_scores = _load_existing_scores(scores_path)
    # Restrict to samples we actually care about this run (respects --tasks/--limit).
    wanted: set[tuple[str, str]] = {(s.task_name, s.video_idx) for s in samples}
    final_scores = [s for s in final_scores if (s.task_name, s.video_idx) in wanted]

    summary = RunSummary(
        model_name=model_name,
        judge=judge.name,
        timestamp=datetime.now().isoformat(),
        samples=final_scores,
        In_Domain=_aggregate(final_scores, "In_Domain"),
        Out_of_Domain=_aggregate(final_scores, "Out_of_Domain"),
        overall=_aggregate(final_scores, None),
    )
    _write_summary_files(out_dir, summary)

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
