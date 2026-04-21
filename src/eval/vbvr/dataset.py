"""Walk model outputs + GT to produce EvalSample records."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from .types import Domain, EvalSample, Split

_GT_DOMAIN_DIRS: dict[str, Domain] = {
    "In-Domain_50": "In_Domain",
    "Out-of-Domain_50": "Out_of_Domain",
}
_SPLITS: tuple[Split, ...] = ("Open_60", "Hidden_40")


def _build_task_to_domain(gt_base: Path) -> dict[str, tuple[Domain, Path]]:
    """task_name -> (domain_label, gt_task_dir)."""
    task_to_domain: dict[str, tuple[Domain, Path]] = {}
    for dir_name, domain in _GT_DOMAIN_DIRS.items():
        domain_path = gt_base / dir_name
        if not domain_path.is_dir():
            logger.warning("GT domain dir missing: {}", domain_path)
            continue
        for task_dir in domain_path.iterdir():
            if task_dir.is_dir():
                task_to_domain[task_dir.name] = (domain, task_dir)
    return task_to_domain


def _find_gt_files(gt_task_dir: Path, video_idx: str) -> tuple[Path, Path, Path, Path | None, str] | None:
    """Return (gt_dir, first_frame, final_frame, gt_video_or_None, prompt) — or None if incomplete."""
    gt_dir = gt_task_dir / video_idx
    if not gt_dir.is_dir():
        return None
    first_frame = gt_dir / "first_frame.png"
    final_frame = gt_dir / "final_frame.png"
    prompt_file = gt_dir / "prompt.txt"
    if not (first_frame.exists() and final_frame.exists() and prompt_file.exists()):
        return None
    gt_video = gt_dir / "ground_truth.mp4"
    return (
        gt_dir,
        first_frame,
        final_frame,
        gt_video if gt_video.exists() else None,
        prompt_file.read_text().strip(),
    )


def discover_samples(
    model_output: Path,
    gt_base: Path,
    tasks: list[str] | None = None,
) -> list[EvalSample]:
    """
    Walk `<model_output>/{Open_60,Hidden_40}/<task>/<idx>.mp4` and produce
    EvalSample records by matching each generated video against its GT.
    Videos without a matching GT record are skipped with a warning.
    """
    task_to_domain = _build_task_to_domain(gt_base)
    task_filter = set(tasks) if tasks else None

    samples: list[EvalSample] = []
    for split in _SPLITS:
        split_path = model_output / split
        if not split_path.is_dir():
            continue
        for task_dir in sorted(split_path.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            task_name = task_dir.name
            if task_filter is not None and task_name not in task_filter:
                continue
            gt_info = task_to_domain.get(task_name)
            if gt_info is None:
                logger.warning("No GT for task {}; skipping", task_name)
                continue
            domain, gt_task_dir = gt_info

            for video_file in sorted(task_dir.iterdir()):
                if video_file.suffix != ".mp4":
                    continue
                video_idx = video_file.stem
                gt_files = _find_gt_files(gt_task_dir, video_idx)
                if gt_files is None:
                    logger.warning("GT incomplete for {}/{}; skipping", task_name, video_idx)
                    continue
                gt_dir, first_frame, final_frame, gt_video, prompt = gt_files
                samples.append(
                    EvalSample(
                        task_name=task_name,
                        video_idx=video_idx,
                        split=split,
                        domain=domain,
                        video_path=video_file,
                        gt_dir=gt_dir,
                        gt_first_frame=first_frame,
                        gt_final_frame=final_frame,
                        gt_video_path=gt_video,
                        prompt=prompt,
                    )
                )

    logger.info("Discovered {} samples across {} tasks", len(samples), len({s.task_name for s in samples}))
    return samples
