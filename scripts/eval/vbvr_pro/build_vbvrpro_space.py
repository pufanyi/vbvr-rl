#!/usr/bin/env python3
"""Build the static Hugging Face Space for the strict VBVR-Pro sweep.

The generated tree is intentionally placed under ignored ``storage/``. Scored
videos are hard-linked when possible so the local build does not duplicate
roughly 4 GiB of media before upload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP_ROOT = REPO_ROOT / "storage/eval_out/vbvr_pro_main_v2_indomain_strict_manifest_326f7bda"
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT / "storage/eval_out/vbvr_pro_main_v2" / "sft_vbvr_5b_256x256x161_full_lr_1e-5_checkpoint-epoch1"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "storage/hf_spaces/vbvrpro_output"
TEMPLATE_ROOT = Path(__file__).with_name("vbvrpro_space")
SCORED_VIDEO_DIR = "eval_1024x1024_161f_5s"
SCORE_FILE = "scores/eval_1024x1024_161f_5s_vbvr_results.json"
MANIFEST_SHA256 = "326f7bda3743e9c66dc0c29445661a5dda4ad0cee4cb8838c3fcfd0c4a149deb"
EVALKIT_REVISION = "42a1593d8e493370c768be8e43646f0e0a9d8525"

RUN_PATTERN = re.compile(
    r"^dancegrpo_vbvr_pro_5b_checkpoint-(?P<checkpoint>\d+)"
    r"(?P<suffix>-euler|-cps-noise-0\.(?:3|7))?$"
)


@dataclass(frozen=True)
class RunSpec:
    source: Path
    run_id: str
    label: str
    short_label: str
    checkpoint: int | None
    mode: str
    mode_label: str
    mode_order: int
    is_baseline: bool = False


MODE_BY_SUFFIX = {
    "": ("unipc", "UniPC ODE", 0),
    "-euler": ("euler", "Euler ODE", 1),
    "-cps-noise-0.3": ("cps03", "CPS 0.3", 2),
    "-cps-noise-0.7": ("cps07", "CPS 0.7", 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of hard-linking them into the generated Space.",
    )
    parser.add_argument(
        "--video-url-prefix",
        help=(
            "Optional absolute URL prefix for externally hosted videos. The run ID and scored-video-relative path "
            "are appended to this prefix."
        ),
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Do not materialize videos in the Space tree. Requires --video-url-prefix.",
    )
    args = parser.parse_args()
    if args.skip_videos and not args.video_url_prefix:
        parser.error("--skip-videos requires --video-url-prefix")
    return args


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(path)


def discover_runs(sweep_root: Path, baseline_root: Path) -> list[RunSpec]:
    runs = [
        RunSpec(
            source=baseline_root,
            run_id="baseline",
            label="SFT epoch-1 baseline",
            short_label="SFT baseline",
            checkpoint=None,
            mode="baseline",
            mode_label="SFT baseline",
            mode_order=-1,
            is_baseline=True,
        )
    ]
    for path in sweep_root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected run directory: {path}")
        checkpoint = int(match.group("checkpoint"))
        suffix = match.group("suffix") or ""
        mode, mode_label, mode_order = MODE_BY_SUFFIX[suffix]
        runs.append(
            RunSpec(
                source=path,
                run_id=f"ckpt-{checkpoint}-{mode}",
                label=f"Checkpoint {checkpoint} · {mode_label}",
                short_label=f"Step {checkpoint} · {mode_label}",
                checkpoint=checkpoint,
                mode=mode,
                mode_label=mode_label,
                mode_order=mode_order,
            )
        )
    return [runs[0], *sorted(runs[1:], key=lambda run: (run.checkpoint or 0, run.mode_order))]


def summary_scores(result: dict[str, Any]) -> dict[str, float]:
    summary = result["summary"]
    return {
        "overall": float(summary["overall"]["mean_score"]),
        "inDomain": float(summary["In_Domain"]["mean_score"]),
        "outOfDomain": float(summary["Out_of_Domain"]["mean_score"]),
    }


def task_scores(result: dict[str, Any]) -> dict[str, float]:
    return {name: float(score) for name, score in result["summary"]["overall"]["by_task"].items()}


def sample_key(sample: dict[str, Any]) -> str:
    return "|".join((sample["split"], sample["task_name"], sample["video_file"]))


def checked_samples(result: dict[str, Any], run: RunSpec) -> list[dict[str, Any]]:
    samples = result["samples"]
    if len(samples) != 500:
        raise ValueError(f"{run.source}: expected 500 score samples, found {len(samples)}")
    errors = [sample for sample in samples if sample.get("error") is not None]
    if errors:
        raise ValueError(f"{run.source}: found {len(errors)} scorer errors")
    keys = [sample_key(sample) for sample in samples]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{run.source}: duplicate scored sample keys")
    return samples


def build_task_catalog(
    baseline_root: Path,
    baseline_samples: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    manifest = read_json(baseline_root / "eval_samples.json")
    prompt_by_key = {
        "|".join((item["domain"], item["task_name"], f"{item['video_idx']}.mp4")): item["prompt"] for item in manifest
    }
    catalog: dict[str, dict[str, str]] = {}
    for sample in baseline_samples:
        key = sample_key(sample)
        if key not in prompt_by_key:
            raise ValueError(f"Baseline eval_samples.json has no prompt for {key}")
        existing = catalog.get(sample["task_name"])
        entry = {
            "name": sample["task_name"],
            "domain": sample["split"],
            "category": sample["category"],
        }
        if existing is not None and existing != entry:
            raise ValueError(f"Inconsistent task metadata for {sample['task_name']}")
        catalog[sample["task_name"]] = entry
    if len(catalog) != 100:
        raise ValueError(f"Expected 100 tasks, found {len(catalog)}")
    return sorted(catalog.values(), key=lambda task: task["name"]), prompt_by_key


def materialize_file(source: Path, destination: Path, *, copy: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if not copy and source_stat.st_dev == destination_stat.st_dev and source_stat.st_ino == destination_stat.st_ino:
            return
        if source_stat.st_size == destination_stat.st_size:
            return
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_templates(output_root: Path) -> None:
    for source in TEMPLATE_ROOT.iterdir():
        if source.is_file():
            shutil.copy2(source, output_root / source.name)


def sample_payload(
    sample: dict[str, Any],
    *,
    run: RunSpec,
    baseline_by_key: dict[str, dict[str, Any]],
    prompt_by_key: dict[str, str],
    output_root: Path,
    copy_videos: bool,
    materialize_videos: bool,
    video_url_prefix: str | None,
) -> dict[str, Any]:
    key = sample_key(sample)
    baseline_sample = baseline_by_key[key]
    source_video = Path(sample["video_path"])
    expected_root = run.source / SCORED_VIDEO_DIR
    try:
        relative_video = source_video.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(f"{source_video} is not inside {expected_root}") from error
    if materialize_videos:
        destination = output_root / "videos" / run.run_id / relative_video
        materialize_file(source_video, destination, copy=copy_videos)
        video_url = destination.relative_to(output_root).as_posix()
    else:
        assert video_url_prefix is not None
        video_url = f"{video_url_prefix.rstrip('/')}/{run.run_id}/{relative_video.as_posix()}"
    payload = {
        "id": key,
        "taskName": sample["task_name"],
        "split": sample["split"],
        "category": sample["category"],
        "folder": sample["folder"],
        "videoFile": sample["video_file"],
        "score": float(sample["score"]),
        "baselineScore": float(baseline_sample["score"]),
        "delta": float(sample["score"]) - float(baseline_sample["score"]),
        "dimensions": sample.get("dimensions", {}),
        "videoUrl": video_url,
    }
    if run.is_baseline:
        payload["prompt"] = prompt_by_key[key]
    return payload


def validate_output(
    *,
    output_root: Path,
    run_payloads: list[dict[str, Any]],
    task_catalog: list[dict[str, str]],
    materialize_videos: bool,
) -> None:
    expected_videos = sum(len(payload["samples"]) for payload in run_payloads)
    if materialize_videos:
        output_videos = list((output_root / "videos").rglob("*.mp4"))
        if len(output_videos) != expected_videos:
            raise ValueError(f"Output contains {len(output_videos)} videos; expected {expected_videos}")
        missing = [
            sample["videoUrl"]
            for payload in run_payloads
            for sample in payload["samples"]
            if not (output_root / sample["videoUrl"]).is_file()
        ]
        if missing:
            raise ValueError(f"Output has {len(missing)} missing linked videos")
    elif any(
        not sample["videoUrl"].startswith(("http://", "https://"))
        for payload in run_payloads
        for sample in payload["samples"]
    ):
        raise ValueError("Externally hosted video URLs must be absolute HTTP(S) URLs")
    if len(task_catalog) != 100:
        raise ValueError("Task catalog is incomplete")


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output_root = args.output_root.resolve()
    materialize_videos = not args.skip_videos
    output_root.mkdir(parents=True, exist_ok=True)
    copy_templates(output_root)

    runs = discover_runs(sweep_root, baseline_root)
    if len(runs) != 21:
        raise ValueError(f"Expected baseline plus 20 sweep runs, found {len(runs)} total")

    results = {run.run_id: read_json(run.source / SCORE_FILE) for run in runs}
    samples_by_run = {run.run_id: checked_samples(results[run.run_id], run) for run in runs}
    baseline_samples = samples_by_run["baseline"]
    baseline_by_key = {sample_key(sample): sample for sample in baseline_samples}
    baseline_keys = set(baseline_by_key)
    for run in runs[1:]:
        keys = {sample_key(sample) for sample in samples_by_run[run.run_id]}
        if keys != baseline_keys:
            missing = baseline_keys - keys
            extra = keys - baseline_keys
            raise ValueError(f"{run.source}: sample mismatch (missing={len(missing)}, extra={len(extra)})")

    task_catalog, prompt_by_key = build_task_catalog(baseline_root, baseline_samples)
    baseline_task_scores = task_scores(results["baseline"])
    run_payloads: list[dict[str, Any]] = []
    index_runs: list[dict[str, Any]] = []

    for run in runs:
        print(f"Building {run.run_id} from {run.source}")
        samples = [
            sample_payload(
                sample,
                run=run,
                baseline_by_key=baseline_by_key,
                prompt_by_key=prompt_by_key,
                output_root=output_root,
                copy_videos=args.copy_videos,
                materialize_videos=materialize_videos,
                video_url_prefix=args.video_url_prefix,
            )
            for sample in samples_by_run[run.run_id]
        ]
        run_task_scores = task_scores(results[run.run_id])
        deltas = {name: run_task_scores[name] - baseline_task_scores[name] for name in baseline_task_scores}
        task_stats = {
            "wins": sum(delta > 1e-10 for delta in deltas.values()),
            "losses": sum(delta < -1e-10 for delta in deltas.values()),
            "ties": sum(abs(delta) <= 1e-10 for delta in deltas.values()),
        }
        data_url = f"data/runs/{run.run_id}.json"
        source_result_url = f"data/source_results/{run.run_id}.json"
        payload = {
            "id": run.run_id,
            "label": run.label,
            "samples": samples,
        }
        run_payloads.append(payload)
        write_json(output_root / data_url, payload)
        source_result_path = output_root / source_result_url
        source_result_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run.source / SCORE_FILE, source_result_path)
        index_runs.append(
            {
                "id": run.run_id,
                "label": run.label,
                "shortLabel": run.short_label,
                "checkpoint": run.checkpoint,
                "mode": run.mode,
                "modeLabel": run.mode_label,
                "modeOrder": run.mode_order,
                "isBaseline": run.is_baseline,
                "scores": summary_scores(results[run.run_id]),
                "taskScores": run_task_scores,
                "taskStats": task_stats,
                "dataUrl": data_url,
                "sourceResultUrl": source_result_url,
            }
        )

    validate_output(
        output_root=output_root,
        run_payloads=run_payloads,
        task_catalog=task_catalog,
        materialize_videos=materialize_videos,
    )
    total_video_bytes = sum(
        Path(sample["video_path"]).stat().st_size for run in runs for sample in samples_by_run[run.run_id]
    )
    baseline_index = index_runs[0]
    index = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "title": "VBVR-Pro Output Explorer",
        "manifestSha256": MANIFEST_SHA256,
        "evalkitRevision": EVALKIT_REVISION,
        "videoUrlPrefix": args.video_url_prefix,
        "runCount": len(runs) - 1,
        "taskCount": len(task_catalog),
        "sampleCountPerRun": len(baseline_samples),
        "totalVideoCount": sum(len(payload["samples"]) for payload in run_payloads),
        "totalVideoBytes": total_video_bytes,
        "modeOrder": ["unipc", "euler", "cps03", "cps07"],
        "baseline": baseline_index,
        "runs": index_runs,
        "tasks": task_catalog,
    }
    write_json(output_root / "data/index.json", index)
    print(
        f"Built {output_root}: {index['totalVideoCount']} indexed videos, "
        f"{total_video_bytes / 1024**3:.2f} GiB source media, {len(task_catalog)} tasks, "
        f"materialized={materialize_videos}"
    )


if __name__ == "__main__":
    main()
