#!/usr/bin/env python3
"""Build the static VBVR-Pro sampler-trajectory comparison Space.

The source tree contains 30 native trajectory videos, one optional overview
grid, and one full-resolution final video per sample/cell. The generated Space
keeps only a compact JSON index and streams media from companion Hugging Face
Datasets in deployment. Local builds hard-link media by default so they do not
duplicate the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAJECTORY_ROOT = REPO_ROOT / "storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "storage/hf_spaces/vbvrpro_sampler_trajectories"
TEMPLATE_ROOT = Path(__file__).with_name("vbvr_trajectory_space")
DATASET_TEMPLATE_ROOT = Path(__file__).with_name("vbvr_trajectory_dataset")
STEP_COUNT = 30


@dataclass(frozen=True)
class ModelSpec:
    id: str
    source_prefix: str
    label: str
    short_label: str
    description: str


@dataclass(frozen=True)
class SamplerSpec:
    id: str
    source_suffix: str
    label: str
    short_label: str
    family: str
    noise_scale: float | None = None


@dataclass(frozen=True)
class CellSpec:
    id: str
    source: Path
    model: ModelSpec
    sampler: SamplerSpec


@dataclass(frozen=True)
class CellScores:
    scores: list[float]
    summary: dict[str, Any]
    contract: dict[str, str]


MODELS = (
    ModelSpec(
        id="baseline",
        source_prefix="baseline",
        label="DiffSynth step-35500 baseline",
        short_label="Baseline",
        description="The pre-RL DiffSynth checkpoint used to initialize DanceGRPO.",
    ),
    ModelSpec(
        id="checkpoint-2200",
        source_prefix="2200",
        label="DanceGRPO checkpoint 2200",
        short_label="Step 2200",
        description="The DanceGRPO model after 2,200 optimizer updates.",
    ),
)

SAMPLERS = (
    SamplerSpec("cps-0.1", "cps0p1", "Flow-CPS · noise 0.1", "CPS 0.1", "Flow-CPS", 0.1),
    SamplerSpec("cps-0.3", "cps0p3", "Flow-CPS · noise 0.3", "CPS 0.3", "Flow-CPS", 0.3),
    SamplerSpec("cps-0.7", "cps0p7", "Flow-CPS · noise 0.7", "CPS 0.7", "Flow-CPS", 0.7),
    SamplerSpec("cps-0.9", "cps0p9", "Flow-CPS · noise 0.9", "CPS 0.9", "Flow-CPS", 0.9),
    SamplerSpec("euler", "euler", "FlowMatch Euler ODE", "Euler ODE", "ODE"),
    SamplerSpec("unipc", "unipc", "UniPC ODE", "UniPC ODE", "ODE"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset-output-root",
        type=Path,
        help="Optional companion Dataset staging root. Media is materialized there instead of inside the Space.",
    )
    parser.add_argument(
        "--media-url-prefix",
        help=(
            "URL prefix ending immediately before the cell ID. For deployment this is normally "
            "https://huggingface.co/datasets/OWNER/REPO/resolve/main/videos"
        ),
    )
    parser.add_argument(
        "--step-dataset-output-root",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help=(
            "Repeat once per model to stage native step videos in separate Dataset trees, "
            "for example baseline=storage/hf_datasets/...-baseline-steps."
        ),
    )
    parser.add_argument(
        "--step-media-url-prefix",
        action="append",
        default=[],
        metavar="MODEL=URL",
        help=(
            "Repeat once per model with the deployed native-step Dataset URL prefix ending "
            "immediately before the cell ID."
        ),
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Build only the frontend/index. Requires --media-url-prefix and cannot be combined with Dataset staging.",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy media instead of hard-linking it into the generated local/Dataset tree.",
    )
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument("--expected-tasks", type=int, default=100)
    args = parser.parse_args()
    if args.skip_videos and not args.media_url_prefix:
        parser.error("--skip-videos requires --media-url-prefix")
    if args.skip_videos and args.dataset_output_root is not None:
        parser.error("--skip-videos cannot be combined with --dataset-output-root")
    if args.skip_videos and args.step_dataset_output_root:
        parser.error("--skip-videos cannot be combined with --step-dataset-output-root")
    if args.dataset_output_root is not None and not args.media_url_prefix:
        parser.error("--dataset-output-root requires --media-url-prefix for the generated Space")
    try:
        args.step_dataset_output_roots = parse_model_mapping(
            args.step_dataset_output_root,
            option="--step-dataset-output-root",
            convert=Path,
        )
        args.step_media_url_prefixes = parse_model_mapping(
            args.step_media_url_prefix,
            option="--step-media-url-prefix",
            convert=str,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.step_dataset_output_roots and not args.step_media_url_prefixes:
        parser.error("--step-dataset-output-root requires --step-media-url-prefix for every model")
    if args.expected_samples <= 0 or args.expected_tasks <= 0:
        parser.error("expected counts must be positive")
    return args


def parse_model_mapping(values: list[str], *, option: str, convert: Any) -> dict[str, Any]:
    expected = {model.id for model in MODELS}
    mapping: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key or not raw:
            raise ValueError(f"{option} expects MODEL=VALUE, got {value!r}")
        if key not in expected:
            raise ValueError(f"{option} has unknown model {key!r}; expected {sorted(expected)}")
        if key in mapping:
            raise ValueError(f"{option} repeats model {key!r}")
        mapping[key] = convert(raw)
    if mapping and set(mapping) != expected:
        raise ValueError(f"{option} must cover exactly {sorted(expected)}")
    return mapping


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_recorded_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def copy_templates(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in source_root.iterdir():
        if source.is_file():
            shutil.copy2(source, destination_root / source.name)


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


def discover_cells(trajectory_root: Path, *, expected_samples: int) -> list[CellSpec]:
    cells: list[CellSpec] = []
    for model in MODELS:
        for sampler in SAMPLERS:
            source = trajectory_root / f"{model.source_prefix}-{sampler.source_suffix}"
            if not source.is_dir():
                raise FileNotFoundError(source)
            cell_manifest_path = source / "cell_manifest.json"
            cell_manifest = read_json(cell_manifest_path)
            if cell_manifest.get("state") != "complete":
                raise ValueError(f"{cell_manifest_path}: cell is not complete")
            completed = int(cell_manifest.get("completed_count", -1))
            declared = int(cell_manifest.get("sample_count", -1))
            if completed != expected_samples or declared != expected_samples:
                raise ValueError(
                    f"{cell_manifest_path}: expected {expected_samples} complete samples, "
                    f"found sample_count={declared}, completed_count={completed}"
                )
            cells.append(
                CellSpec(
                    id=f"{model.id}--{sampler.id}",
                    source=source,
                    model=model,
                    sampler=sampler,
                )
            )
    return cells


def sample_paths(cell: CellSpec, *, expected_samples: int) -> list[Path]:
    samples = sorted(path.parent.relative_to(cell.source) for path in cell.source.rglob("steps_grid.mp4"))
    if len(samples) != expected_samples:
        raise ValueError(f"{cell.source}: expected {expected_samples} trajectory grids, found {len(samples)}")
    if len(set(samples)) != len(samples):
        raise ValueError(f"{cell.source}: duplicate sample paths")
    for relative in samples:
        if len(relative.parts) != 3:
            raise ValueError(f"{cell.source / relative}: expected domain/task/sample layout")
        sample_root = cell.source / relative
        for filename in ("manifest.json", "steps_grid.mp4", "final_00.mp4"):
            if not (sample_root / filename).is_file():
                raise FileNotFoundError(sample_root / filename)
        for step_index in range(STEP_COUNT):
            step_path = sample_root / f"step_{step_index:02d}.mp4"
            if not step_path.is_file():
                raise FileNotFoundError(step_path)
    print(
        f"[discover] {cell.id}: {len(samples)} samples with {STEP_COUNT} native step videos each",
        flush=True,
    )
    return samples


def canonical_sample_payload(cell: CellSpec, relative: Path) -> dict[str, Any]:
    manifest = read_json(cell.source / relative / "manifest.json")
    summary = manifest.get("summary") or {}
    recorded_name = summary.get("name") or manifest.get("sample_name")
    if recorded_name != relative.as_posix():
        raise ValueError(
            f"{cell.source / relative / 'manifest.json'}: sample identity mismatch "
            f"({recorded_name!r} != {relative.as_posix()!r})"
        )
    prompt = summary.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{cell.source / relative / 'manifest.json'}: missing sample prompt")
    domain_folder, task_name, sample_id = relative.parts
    domain = {
        "In-Domain_50": "in-domain",
        "Out-of-Domain_50": "out-of-domain",
    }.get(domain_folder)
    if domain is None:
        raise ValueError(f"Unexpected domain folder: {domain_folder}")
    previews = manifest.get("step_previews")
    if not isinstance(previews, list) or len(previews) != STEP_COUNT:
        raise ValueError(f"{cell.source / relative / 'manifest.json'}: expected {STEP_COUNT} step previews")
    return {
        "id": relative.as_posix(),
        "domain": domain,
        "domainFolder": domain_folder,
        "task": task_name,
        "sample": sample_id,
        "seed": int(manifest.get("seed", 0)),
        "prompt": prompt,
    }


def schedule_payload(cell: CellSpec, relative: Path) -> list[dict[str, Any]]:
    manifest = read_json(cell.source / relative / "manifest.json")
    previews = manifest.get("step_previews")
    if not isinstance(previews, list) or len(previews) != STEP_COUNT:
        raise ValueError(f"{cell.source / relative / 'manifest.json'}: expected {STEP_COUNT} step previews")
    return [
        {
            "step": int(preview["display_step"]),
            "kind": str(preview["kind"]),
            "sourceSigma": float(preview["source_sigma"]),
            "outputSigma": float(preview["output_sigma"]),
        }
        for preview in previews
    ]


def result_sample_id(sample: dict[str, Any], *, result_path: Path) -> str:
    folder = sample.get("folder")
    task = sample.get("task_name")
    video_file = sample.get("video_file")
    if not all(isinstance(value, str) and value for value in (folder, task, video_file)):
        raise ValueError(f"{result_path}: score sample is missing folder/task_name/video_file")
    video_path = Path(video_file)
    if video_path.name != video_file or video_path.suffix.lower() != ".mp4":
        raise ValueError(f"{result_path}: unexpected score video_file {video_file!r}")
    expected_split = {
        "In-Domain_50": "In_Domain",
        "Out-of-Domain_50": "Out_of_Domain",
    }.get(folder)
    if expected_split is None or sample.get("split") != expected_split:
        raise ValueError(f"{result_path}: inconsistent score domain for {folder!r}")
    return Path(folder, task, video_path.stem).as_posix()


def load_cell_scores(cell: CellSpec, canonical_paths: list[Path]) -> CellScores:
    cell_manifest_path = cell.source / "cell_manifest.json"
    cell_manifest = read_json(cell_manifest_path)
    formal_final_root = resolve_recorded_path(
        cell_manifest.get("formal_final_root"),
        field=f"{cell_manifest_path}: formal_final_root",
    )
    if not formal_final_root.is_dir():
        raise FileNotFoundError(formal_final_root)
    formal_run_root = formal_final_root.parent

    result_files = sorted((formal_run_root / "scores").glob("*_vbvr_results.json"))
    if len(result_files) != 1:
        raise ValueError(f"{formal_run_root}: expected exactly one *_vbvr_results.json, found {len(result_files)}")
    result_path = result_files[0]
    provenance_path = formal_run_root / "score-provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = read_json(provenance_path)
    values = provenance.get("values") or {}
    if provenance.get("stage") != "vbvr-pro-score" or values.get("state") != "complete":
        raise ValueError(f"{provenance_path}: score stage is not complete")

    result_record = (provenance.get("output_files") or {}).get("result") or {}
    recorded_result_path = resolve_recorded_path(
        result_record.get("path"),
        field=f"{provenance_path}: output_files.result.path",
    )
    if recorded_result_path != result_path.resolve():
        raise ValueError(f"{provenance_path}: result path does not match {result_path}")
    result_sha256 = sha256_file(result_path)
    if result_record.get("sha256") != result_sha256:
        raise ValueError(f"{provenance_path}: result SHA-256 does not match {result_path}")
    if int(result_record.get("size", -1)) != result_path.stat().st_size:
        raise ValueError(f"{provenance_path}: result size does not match {result_path}")

    evalkit_revision = values.get("evalkit_revision")
    evalkit_revision_actual = values.get("evalkit_revision_actual")
    evalkit_source_sha256 = values.get("evalkit_source_sha256")
    # Copied EvalKit snapshots may not retain Git metadata and explicitly record
    # "unavailable" here; the pinned revision plus source-tree digest remain required.
    if not isinstance(evalkit_revision, str) or evalkit_revision_actual not in {
        evalkit_revision,
        "unavailable",
    }:
        raise ValueError(f"{provenance_path}: EvalKit revision is missing or inconsistent")
    if not isinstance(evalkit_source_sha256, str) or len(evalkit_source_sha256) != 64:
        raise ValueError(f"{provenance_path}: invalid EvalKit source SHA-256")
    dependency_value = values.get("scorer_dependencies")
    try:
        dependencies = json.loads(dependency_value) if isinstance(dependency_value, str) else dependency_value
    except json.JSONDecodeError as error:
        raise ValueError(f"{provenance_path}: invalid scorer_dependencies JSON") from error
    if not isinstance(dependencies, dict):
        raise ValueError(f"{provenance_path}: missing scorer dependency contract")
    dependency_contract = dependencies.get("contract")
    dependency_sha256 = dependencies.get("sha256")
    if not isinstance(dependency_contract, str) or not dependency_contract:
        raise ValueError(f"{provenance_path}: missing scorer dependency contract name")
    if not isinstance(dependency_sha256, str) or len(dependency_sha256) != 64:
        raise ValueError(f"{provenance_path}: invalid scorer dependency SHA-256")

    canonical_ids = [relative.as_posix() for relative in canonical_paths]
    canonical_set = set(canonical_ids)
    result = read_json(result_path)
    result_samples = result.get("samples")
    if not isinstance(result_samples, list) or len(result_samples) != len(canonical_paths):
        found = len(result_samples) if isinstance(result_samples, list) else "invalid"
        raise ValueError(f"{result_path}: expected {len(canonical_paths)} score samples, found {found}")

    score_by_id: dict[str, float] = {}
    error_count = 0
    for sample in result_samples:
        if not isinstance(sample, dict):
            raise ValueError(f"{result_path}: score sample must be an object")
        sample_id = result_sample_id(sample, result_path=result_path)
        if sample_id in score_by_id:
            raise ValueError(f"{result_path}: duplicate score sample {sample_id}")
        raw_score = sample.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError(f"{result_path}: non-numeric score for {sample_id}")
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{result_path}: score outside [0, 1] for {sample_id}: {score}")
        task_specific = (sample.get("dimensions") or {}).get("task_specific")
        if not isinstance(task_specific, (int, float)) or not math.isclose(
            float(task_specific), score, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{result_path}: task_specific score mismatch for {sample_id}")
        score_by_id[sample_id] = score
        error_count += sample.get("error") is not None
    if set(score_by_id) != canonical_set:
        raise ValueError(
            f"{result_path}: score sample mismatch "
            f"(missing={len(canonical_set - set(score_by_id))}, extra={len(set(score_by_id) - canonical_set)})"
        )

    for relative in canonical_paths:
        sample_manifest_path = cell.source / relative / "manifest.json"
        sample_manifest = read_json(sample_manifest_path)
        binding = sample_manifest.get("formal_final_binding") or {}
        expected_source = (formal_final_root / relative.parent / f"{relative.name}.mp4").resolve()
        if not expected_source.is_file():
            raise FileNotFoundError(expected_source)
        bound_source = resolve_recorded_path(
            binding.get("source"),
            field=f"{sample_manifest_path}: formal_final_binding.source",
        )
        if bound_source != expected_source:
            raise ValueError(f"{sample_manifest_path}: formal final binding source mismatch")
        binding_sha256 = binding.get("sha256")
        if not isinstance(binding_sha256, str) or len(binding_sha256) != 64:
            raise ValueError(f"{sample_manifest_path}: invalid formal final binding SHA-256")
        bound_outputs = binding.get("bound_outputs")
        if not isinstance(bound_outputs, list):
            raise ValueError(f"{sample_manifest_path}: missing formal final bound outputs")
        resolved_outputs = {
            resolve_recorded_path(output, field=f"{sample_manifest_path}: formal final bound output")
            for output in bound_outputs
        }
        expected_outputs = {
            (cell.source / relative / "final_00.mp4").resolve(),
            (cell.source / relative / "step_29.mp4").resolve(),
        }
        if not expected_outputs.issubset(resolved_outputs):
            raise ValueError(f"{sample_manifest_path}: final_00.mp4 and step_29.mp4 are not formally bound")

    aligned_scores = [score_by_id[sample_id] for sample_id in canonical_ids]
    summary = result.get("summary") or {}
    summary_groups = {
        "In_Domain": [
            score
            for relative, score in zip(canonical_paths, aligned_scores, strict=True)
            if relative.parts[0] == "In-Domain_50"
        ],
        "Out_of_Domain": [
            score
            for relative, score in zip(canonical_paths, aligned_scores, strict=True)
            if relative.parts[0] == "Out-of-Domain_50"
        ],
        "overall": aligned_scores,
    }
    for group_name, group_scores in summary_groups.items():
        group_summary = summary.get(group_name) or {}
        expected_mean = math.fsum(group_scores) / len(group_scores)
        if int(group_summary.get("num_samples", -1)) != len(group_scores) or not math.isclose(
            float(group_summary.get("mean_score", math.nan)),
            expected_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{result_path}: inconsistent {group_name} score summary")

    score_summary = {
        "overall": float(summary["overall"]["mean_score"]),
        "inDomain": float(summary["In_Domain"]["mean_score"]),
        "outOfDomain": float(summary["Out_of_Domain"]["mean_score"]),
        "sampleCount": len(aligned_scores),
        "errorCount": error_count,
        "resultSha256": result_sha256,
    }
    contract = {
        "evalkitRevision": evalkit_revision,
        "evalkitSourceSha256": evalkit_source_sha256,
        "dependencyContract": dependency_contract,
        "dependencyContractSha256": dependency_sha256,
    }
    print(
        f"[scores] {cell.id}: {len(aligned_scores)} final-output scores, "
        f"mean={score_summary['overall']:.4f}, errors={error_count}",
        flush=True,
    )
    return CellScores(scores=aligned_scores, summary=score_summary, contract=contract)


def media_url(prefix: str, cell: CellSpec, relative: Path, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{cell.id}/{relative.as_posix()}/{filename}"


def build_space(
    *,
    trajectory_root: Path,
    output_root: Path,
    dataset_output_root: Path | None,
    media_url_prefix: str | None,
    step_dataset_output_roots: dict[str, Path] | None = None,
    step_media_url_prefixes: dict[str, str] | None = None,
    skip_videos: bool,
    copy_videos: bool,
    expected_samples: int,
    expected_tasks: int,
) -> dict[str, Any]:
    trajectory_root = trajectory_root.resolve()
    output_root = output_root.resolve()
    dataset_output_root = dataset_output_root.resolve() if dataset_output_root is not None else None
    step_dataset_output_roots = {
        model_id: root.resolve() for model_id, root in (step_dataset_output_roots or {}).items()
    }
    step_media_url_prefixes = dict(step_media_url_prefixes or {})
    model_ids = {model.id for model in MODELS}
    for mapping, label in (
        (step_dataset_output_roots, "step Dataset output roots"),
        (step_media_url_prefixes, "step media URL prefixes"),
    ):
        if mapping and set(mapping) != model_ids:
            raise ValueError(f"{label} must cover exactly {sorted(model_ids)}")
    if step_dataset_output_roots and not step_media_url_prefixes:
        raise ValueError("step Dataset output roots require step media URL prefixes")
    if skip_videos and step_dataset_output_roots:
        raise ValueError("Skipped media cannot be combined with step Dataset output roots")

    copy_templates(TEMPLATE_ROOT, output_root)
    dataset_roots = set(step_dataset_output_roots.values())
    if dataset_output_root is not None:
        dataset_roots.add(dataset_output_root)
    for dataset_root in dataset_roots:
        if dataset_root != output_root:
            copy_templates(DATASET_TEMPLATE_ROOT, dataset_root)

    cells = discover_cells(trajectory_root, expected_samples=expected_samples)
    paths_by_cell = {cell.id: sample_paths(cell, expected_samples=expected_samples) for cell in cells}
    canonical_cell = cells[0]
    canonical_paths = paths_by_cell[canonical_cell.id]
    canonical_set = set(canonical_paths)
    for cell in cells[1:]:
        paths = set(paths_by_cell[cell.id])
        if paths != canonical_set:
            raise ValueError(
                f"{cell.source}: sample mismatch against {canonical_cell.source} "
                f"(missing={len(canonical_set - paths)}, extra={len(paths - canonical_set)})"
            )

    samples = [canonical_sample_payload(canonical_cell, relative) for relative in canonical_paths]
    task_counts = Counter((sample["domain"], sample["task"]) for sample in samples)
    if len(task_counts) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} tasks, found {len(task_counts)}")
    samples_per_task = set(task_counts.values())
    if len(samples_per_task) != 1:
        raise ValueError(f"Task sample counts are not uniform: {sorted(samples_per_task)}")

    cell_scores = {cell.id: load_cell_scores(cell, canonical_paths) for cell in cells}
    score_contract = cell_scores[cells[0].id].contract
    for cell in cells[1:]:
        if cell_scores[cell.id].contract != score_contract:
            raise ValueError(f"{cell.source}: score contract differs from {cells[0].source}")

    first_relative = canonical_paths[0]
    schedules: dict[str, list[dict[str, Any]]] = {}
    for sampler in SAMPLERS:
        baseline_cell = next(cell for cell in cells if cell.model.id == "baseline" and cell.sampler.id == sampler.id)
        trained_cell = next(
            cell for cell in cells if cell.model.id == "checkpoint-2200" and cell.sampler.id == sampler.id
        )
        baseline_schedule = schedule_payload(baseline_cell, first_relative)
        trained_schedule = schedule_payload(trained_cell, first_relative)
        if baseline_schedule != trained_schedule:
            raise ValueError(f"Sampler schedule differs between baseline and checkpoint 2200: {sampler.id}")
        schedules[sampler.id] = baseline_schedule

    if skip_videos:
        overview_materialize_root = None
        effective_prefix = media_url_prefix
    elif dataset_output_root is not None:
        overview_materialize_root = dataset_output_root
        effective_prefix = media_url_prefix
    else:
        overview_materialize_root = output_root
        effective_prefix = media_url_prefix or "videos"
    assert effective_prefix is not None
    effective_step_prefixes = step_media_url_prefixes or {model.id: effective_prefix for model in MODELS}
    if skip_videos:
        step_materialize_roots: dict[str, Path | None] = {model.id: None for model in MODELS}
    elif step_dataset_output_roots:
        step_materialize_roots = dict(step_dataset_output_roots)
    else:
        step_materialize_roots = {model.id: overview_materialize_root for model in MODELS}

    overview_media_bytes = 0
    original_step_media_bytes = 0
    expected_media_counts: Counter[Path] = Counter()
    for cell in cells:
        for relative in canonical_paths:
            for filename in ("steps_grid.mp4", "final_00.mp4"):
                source = cell.source / relative / filename
                overview_media_bytes += source.stat().st_size
                if overview_materialize_root is not None:
                    destination = overview_materialize_root / "videos" / cell.id / relative / filename
                    materialize_file(source, destination, copy=copy_videos)
                    expected_media_counts[overview_materialize_root] += 1
            step_materialize_root = step_materialize_roots[cell.model.id]
            for step_index in range(STEP_COUNT):
                filename = f"step_{step_index:02d}.mp4"
                source = cell.source / relative / filename
                original_step_media_bytes += source.stat().st_size
                if step_materialize_root is not None:
                    destination = step_materialize_root / "videos" / cell.id / relative / filename
                    materialize_file(source, destination, copy=copy_videos)
                    expected_media_counts[step_materialize_root] += 1
        print(
            f"[media] {cell.id}: {len(canonical_paths)} samples, "
            f"{len(canonical_paths) * STEP_COUNT} native step videos",
            flush=True,
        )

    cell_payloads = []
    for cell in cells:
        cell_payloads.append(
            {
                "id": cell.id,
                "model": cell.model.id,
                "sampler": cell.sampler.id,
                "label": f"{cell.model.short_label} · {cell.sampler.short_label}",
                "scoreSummary": cell_scores[cell.id].summary,
            }
        )
    overview_video_count = len(cells) * len(samples) * 2
    original_step_video_count = len(cells) * len(samples) * STEP_COUNT
    total_media_bytes = overview_media_bytes + original_step_media_bytes
    index = {
        "schemaVersion": 3,
        "generatedAt": datetime.now(UTC).isoformat(),
        "title": "VBVR-Pro Sampler Trajectory Compare",
        "sourceDirectory": trajectory_root.name,
        "mediaUrlPrefix": effective_prefix,
        "stepMediaUrlPrefixes": effective_step_prefixes,
        "sampleCount": len(samples),
        "taskCount": len(task_counts),
        "samplesPerTask": next(iter(samples_per_task)),
        "cellCount": len(cells),
        "videoCount": overview_video_count + original_step_video_count,
        "overviewVideoCount": overview_video_count,
        "originalStepVideoCount": original_step_video_count,
        "totalMediaBytes": total_media_bytes,
        "overviewMediaBytes": overview_media_bytes,
        "originalStepMediaBytes": original_step_media_bytes,
        "stepCount": STEP_COUNT,
        "models": [
            {
                "id": model.id,
                "label": model.label,
                "shortLabel": model.short_label,
                "description": model.description,
            }
            for model in MODELS
        ],
        "samplers": [
            {
                "id": sampler.id,
                "label": sampler.label,
                "shortLabel": sampler.short_label,
                "family": sampler.family,
                "noiseScale": sampler.noise_scale,
                "schedule": schedules[sampler.id],
            }
            for sampler in SAMPLERS
        ],
        "cells": cell_payloads,
        "samples": samples,
        "scores": {cell.id: cell_scores[cell.id].scores for cell in cells},
        "scoreContract": {
            "label": "Final EvalKit score",
            "scaleMin": 0.0,
            "scaleMax": 1.0,
            "displayPrecision": 4,
            "scoreCount": len(cells) * len(samples),
            "appliesTo": "final-only",
            "evaluatedArtifact": (
                "The formal final output bound to public step 30 and final_00.mp4, "
                "prepared at 1024×1024×81 frames and 16 FPS for EvalKit scoring."
            ),
            **score_contract,
        },
        "media": {
            "stepFilenamePrefix": "step_",
            "stepFilenameExtension": ".mp4",
            "gridFilename": "steps_grid.mp4",
            "finalFilename": "final_00.mp4",
            "stepDescription": "The native 512×512 video for one selected diffusion step.",
            "gridDescription": "A compressed 6×5 overview of all 30 previews; use Original step for inspection.",
            "finalDescription": "The dedicated 512×512 final output at sigma zero.",
        },
        "trajectorySemantics": (
            "Steps 1–29 are post-CFG predicted-clean x0 previews at the displayed source sigma; "
            "step 30 is the actual final latent decoded at sigma zero."
        ),
    }
    write_json(output_root / "data/index.json", index)
    for dataset_root in dataset_roots:
        write_json(dataset_root / "data/index.json", index)

    for materialize_root, expected_count in expected_media_counts.items():
        actual_count = sum(1 for _ in (materialize_root / "videos").rglob("*.mp4"))
        if actual_count != expected_count:
            raise ValueError(f"{materialize_root}: expected {expected_count} materialized videos, found {actual_count}")
    if skip_videos:
        prefixes = [effective_prefix, *effective_step_prefixes.values()]
        if any(not prefix.startswith(("http://", "https://")) for prefix in prefixes):
            raise ValueError("Skipped media requires absolute HTTP(S) URL prefixes")
    return index


def main() -> None:
    args = parse_args()
    index = build_space(
        trajectory_root=args.trajectory_root,
        output_root=args.output_root,
        dataset_output_root=args.dataset_output_root,
        media_url_prefix=args.media_url_prefix,
        step_dataset_output_roots=args.step_dataset_output_roots,
        step_media_url_prefixes=args.step_media_url_prefixes,
        skip_videos=args.skip_videos,
        copy_videos=args.copy_videos,
        expected_samples=args.expected_samples,
        expected_tasks=args.expected_tasks,
    )
    print(
        f"Built {args.output_root.resolve()}: {index['cellCount']} cells, {index['sampleCount']} samples, "
        f"{index['videoCount']} videos ({index['totalMediaBytes'] / 1024**3:.2f} GiB), "
        f"media_prefix={index['mediaUrlPrefix']}"
    )


if __name__ == "__main__":
    main()
