"""Driver that scores all samples and writes VBVR-compatible output JSONs.

Resumable and data-parallel:

* Each rank appends its SampleScores to ``scores.rank{N}.jsonl`` as soon as
  they land. A killed run picks up where it left off — from any rank — on
  the next invocation.
* Rank ``R`` handles ``samples[R::world_size]`` (round-robin); resume state
  is computed across ALL shards so if one rank was slower, its unfinished
  work is still skipped on restart as long as any other rank hadn't done it.
* Rank 0 shows a single overall progress bar, polling all shards between
  its own samples. Other ranks are silent.
* Only rank 0 writes ``eval_results.json`` + ``summary.json`` (after a
  ``dist.barrier()`` if multi-rank).
"""

from __future__ import annotations

import json
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


def _load_all_shards(out_dir: Path) -> list[SampleScore]:
    """Read every ``scores.rank*.jsonl`` shard; order is stable by rank then line."""
    scores: list[SampleScore] = []
    for shard in sorted(out_dir.glob("scores.rank*.jsonl")):
        scores.extend(_load_existing_scores(shard))
    return scores


def _count_shard_lines(out_dir: Path) -> int:
    """Cheap approximation of total completed samples across all shards."""
    n = 0
    for shard in out_dir.glob("scores.rank*.jsonl"):
        try:
            with shard.open("rb") as f:
                n += sum(1 for _ in f)
        except FileNotFoundError:
            pass
    return n


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
    rank: int = 0,
    world_size: int = 1,
    barrier_fn=None,
) -> RunSummary | None:
    """Score every (task, video) pair; resumable via per-rank JSONL shards.

    Output layout:
      <output_dir>/<model_name>/scores.rank{N}.jsonl  — one shard per rank, append-only
      <output_dir>/<model_name>/eval_results.json     — rank-0-only; full merged run
      <output_dir>/<model_name>/summary.json          — rank-0-only; headline aggregates

    Args:
        rank, world_size: data-parallel sharding via ``samples[rank::world_size]``.
        barrier_fn: optional zero-arg callable to synchronize ranks before
            rank-0 aggregates (e.g. ``torch.distributed.barrier``).
        fresh: rank 0 moves all existing shards to ``.bak`` and starts over.
        retry_errors: re-score samples whose cached entry has an error.

    Returns:
        Full RunSummary on rank 0, ``None`` on other ranks.
    """
    model_name = model_output.name
    out_dir = output_dir / model_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        if fresh:
            for shard in out_dir.glob("scores.rank*.jsonl"):
                shard.rename(shard.with_suffix(".jsonl.bak"))
                logger.info("fresh run — backed up {}", shard.name)
    if barrier_fn is not None:
        barrier_fn()

    samples = discover_samples(model_output, gt_base, tasks=tasks)
    if limit is not None:
        samples = samples[:limit]
        if rank == 0:
            logger.info("limit applied: {} samples", len(samples))

    my_samples = samples[rank::world_size]
    all_existing = _load_all_shards(out_dir)  # unified view across shards
    remaining = _filter_remaining(my_samples, all_existing, retry_errors=retry_errors)
    global_done = len(samples) - len(_filter_remaining(samples, all_existing, retry_errors))

    if rank == 0:
        logger.info(
            "world_size={} | total={} done={} remaining_global={} (my_slice={} remaining={})",
            world_size,
            len(samples),
            global_done,
            len(samples) - global_done,
            len(my_samples),
            len(remaining),
        )

    shard_path = out_dir / f"scores.rank{rank}.jsonl"
    pbar = None
    if rank == 0:
        pbar = tqdm(
            total=len(samples),
            initial=global_done,
            desc=f"[{model_name}|{judge.name}]",
            unit="sample",
        )

    with shard_path.open("a", encoding="utf-8") as f:
        try:
            for sample in remaining:
                score = judge.score(sample)
                f.write(score.model_dump_json() + "\n")
                f.flush()
                if pbar is not None:
                    # Sum lines across all shards so the bar reflects global progress.
                    pbar.n = min(len(samples), _count_shard_lines(out_dir))
                    pbar.refresh()
        finally:
            if pbar is not None:
                pbar.close()

    if barrier_fn is not None:
        barrier_fn()

    if rank != 0:
        return None

    # Rank 0: authoritative aggregation from all shards, scoped to this run.
    final_scores = _load_all_shards(out_dir)
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
