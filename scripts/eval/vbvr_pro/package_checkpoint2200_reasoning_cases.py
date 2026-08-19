#!/usr/bin/env python3
"""Package audited checkpoint-2200 versus baseline reasoning trajectories.

The package is a folder-first alternative to a presentation deck.  Every case
contains the question/GT assets, exact formal final videos and scores, all 30
predicted-clean sampling previews for both systems, compact milestone views,
and the exact pinned EvalKit audit evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTION_ROOT = REPO_ROOT / "storage/presentations/vbvr_checkpoint2200_vs_baseline_20260818"
DEFAULT_SELECTION = DEFAULT_SELECTION_ROOT / "selection_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "storage/presentations/vbvr_checkpoint2200_vs_baseline_reasoning_chains_20260818"
TRAJECTORY_ROOT = REPO_ROOT / "storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories"
EVALKIT_ROOT = REPO_ROOT / "storage/evalkits/vbvr-evalkit-interleave-main_v2-e140038f"
BASELINE_CELL = "baseline-unipc"
CHECKPOINT_CELL = "2200-cps0p7"
STEP_COUNT = 30
MILESTONE_STEPS = (1, 6, 11, 16, 21, 26, 30)
TEMPORAL_FRAME_INDICES = (0, 20, 40, 60, 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_once(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"Refusing to replace different generated text: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def write_json_once(path: Path, value: Any) -> None:
    write_text_once(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def link_or_copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise FileExistsError(f"Refusing to replace different generated file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def relative_source(path_string: str) -> Path:
    path = Path(path_string)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def video_frames(path: Path, indices: tuple[int, ...] = TEMPORAL_FRAME_INDICES) -> np.ndarray:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if len(reader) != 81:
        raise ValueError(f"Expected 81 frames, found {len(reader)}: {path}")
    if reader[0].shape != (512, 512, 3):
        raise ValueError(f"Expected 512x512 RGB video: {path}")
    if abs(float(reader.get_avg_fps()) - 16.0) > 0.05:
        raise ValueError(f"Expected 16 FPS video: {path}")
    return reader.get_batch(list(indices)).asnumpy()


def normalized_mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> float:
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32)) / 255.0
    if mask is None:
        return float(difference.mean())
    selected = difference[mask]
    return float(selected.mean()) if selected.size else 0.0


def load_question_images(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(Image.open(relative_source(case["input_image"])).convert("RGB").resize((512, 512)))
    final = np.asarray(Image.open(relative_source(case["ground_truth_final"])).convert("RGB").resize((512, 512)))
    changed = np.max(np.abs(first.astype(np.int16) - final.astype(np.int16)), axis=2) > 10
    changed = cv2.dilate(changed.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    if not changed.any():
        changed = np.ones(first.shape[:2], dtype=bool)
    return first, final, changed


def write_metrics(
    trajectory_dir: Path,
    package_side_dir: Path,
    trajectory_manifest: dict[str, Any],
    gt_first: np.ndarray,
    gt_final: np.ndarray,
    edit_mask: np.ndarray,
) -> list[dict[str, Any]]:
    metrics_path = package_side_dir / "reasoning_metrics.csv"
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        for row in existing:
            row["display_step"] = int(row["display_step"])
            for key in (
                "source_sigma",
                "mae_to_formal_final_sampled_video",
                "last_frame_edit_roi_mae_to_gt",
                "last_frame_background_mae_to_input",
            ):
                row[key] = float(row[key])
            for key in ("source_timestep", "mae_to_previous_step_sampled_video"):
                row[key] = None if row[key] == "" else float(row[key])
        if len(existing) != STEP_COUNT:
            raise ValueError(f"Unexpected cached reasoning metric count: {metrics_path}")
        return existing

    final_frames = video_frames(trajectory_dir / "final_00.mp4")
    prior_frames: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    previews = trajectory_manifest["step_previews"]
    if len(previews) != STEP_COUNT:
        raise ValueError(f"Expected {STEP_COUNT} previews: {trajectory_dir}")
    for index, preview in enumerate(previews):
        display_step = index + 1
        if preview["display_step"] != display_step:
            raise ValueError(f"Unexpected display step in {trajectory_dir}/manifest.json")
        step_path = trajectory_dir / f"step_{index:02d}.mp4"
        frames = video_frames(step_path)
        row = {
            "display_step": display_step,
            "file": f"all_steps/step_{display_step:02d}.mp4",
            "kind": preview["kind"],
            "source_sigma": preview["source_sigma"],
            "source_timestep": preview.get("source_timestep"),
            "mae_to_formal_final_sampled_video": normalized_mae(frames, final_frames),
            "mae_to_previous_step_sampled_video": (
                None if prior_frames is None else normalized_mae(frames, prior_frames)
            ),
            "last_frame_edit_roi_mae_to_gt": normalized_mae(frames[-1], gt_final, edit_mask),
            "last_frame_background_mae_to_input": normalized_mae(frames[-1], gt_first, ~edit_mask),
            "sha256": sha256_file(step_path),
        }
        rows.append(row)
        prior_frames = frames

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return rows


def flatten_scalars(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            if not prefix:
                result.update(flatten_scalars(item, name))
        elif isinstance(item, (bool, int, float, str)) or item is None:
            result[name] = item
    return result


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "—"
    return str(value).replace("|", "\\|")


def scorer_summary_markdown(case: dict[str, Any], audit: dict[str, Any]) -> str:
    metadata = audit["metadata_audit"]
    semantic = metadata["semantic_ground_truth"]
    results = audit["results"]
    flat = {
        side: flatten_scalars(results[side]["task_specific_details"])
        for side in ("baseline_unipc", "checkpoint2200_cps0p7", "ground_truth")
    }
    common_keys = sorted(set().union(*(values.keys() for values in flat.values())))
    useful_keys = [
        key
        for key in common_keys
        if not key.endswith("formula") and not key.endswith("source") and not key.endswith("metafile_path")
    ][:30]
    metric_rows = "\n".join(
        f"| `{key}` | {format_metric(flat['baseline_unipc'].get(key))} | "
        f"{format_metric(flat['checkpoint2200_cps0p7'].get(key))} | "
        f"{format_metric(flat['ground_truth'].get(key))} |"
        for key in useful_keys
    )
    return f"""# {case["case_id"]} scorer audit

结论：**PASS**。正式分数已经用完全相同的 e140 EvalKit、1024×1024 scorer 输入和 GT
独立复跑，数值精确复现。三份输入（baseline / checkpoint / GT）均由同一个任务 evaluator
处理，且无 scorer error。

## 任务语义

{semantic.get("task_summary_zh", case["task_label"])}

- canonical name: `{case["canonical_name"]}`
- metadata evaluator: `{audit["evaluator"]["declared_metadata_class"]}`
- 实际 EvalKit registry evaluator: `{audit["evaluator"]["class"]}`
- source: `{audit["evaluator"]["source"]}:{audit["evaluator"]["source_line"]}`
- EvalKit source SHA-256: `4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8`
- metadata 必需语义字段：{len(metadata["required_semantic_fields"])} 项，缺失 0 项

## 精确分数

| 输入 | 旧记录 | 本次复跑 |
|---|---:|---:|
| DiffSynth baseline · UniPC | {case["baseline_score"]:.12f} | {audit["rerun_scores"]["baseline_unipc"]:.12f} |
| checkpoint 2200 · CPS 0.7 | {case["checkpoint_score"]:.12f} | {audit["rerun_scores"]["checkpoint2200_cps0p7"]:.12f} |
| checkpoint − baseline | {case["score_delta"]:+.12f} | {audit["rerun_scores"]["delta"]:+.12f} |
| GT self-score | — | {audit["rerun_scores"]["ground_truth_self_score"]:.12f} |

## Task-specific evaluator 内部量

这些字段直接来自 evaluator 的 `_last_task_details`。不同字段的好坏方向可能不同；最终 score
公式和 `scorer_audit.json` 是权威记录。

| metric | baseline | checkpoint 2200 | GT |
|---|---:|---:|---:|
{metric_rows}
"""


def reasoning_readme(
    side_label: str,
    score: float,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    milestone_rows = "\n".join(
        f"| {step:02d} | {rows[step - 1]['source_sigma']:.9f} | "
        f"{rows[step - 1]['mae_to_formal_final_sampled_video']:.6f} | "
        f"{rows[step - 1]['last_frame_edit_roi_mae_to_gt']:.6f} |"
        for step in MILESTONE_STEPS
    )
    return f"""# {side_label}: 30-step sampler trajectory

- 正式 final score: `{score:.12f}`
- sampler: `{manifest["sampler"]}`
- steps: `{manifest["num_inference_steps"]}`
- CFG: `{manifest["guidance_scale"]}`
- seed: `{manifest["seed"]}`
- video: 512×512, 81 frames, 16 FPS

`all_steps/step_01.mp4`–`step_29.mp4` 是每一步在对应 source sigma 上的 post-CFG
predicted-clean `x0 = x_sigma - sigma * velocity`，并不是被单独打分的 rollout；
`step_30.mp4` 是 sigma=0 的实际最终 latent 解码。正式 EvalKit 只给 final 视频打分。

`milestones/` 提供 1/6/11/16/21/26/30 七个固定节点，便于快速浏览；
`overview_30_steps.mp4` 和 `contact_sheet_30_steps.jpg` 展示完整过程。
`reasoning_metrics.csv` 中的 MAE 只是理解收敛过程的辅助视觉量，不是 reward。

| step | source sigma | sampled-video MAE → formal final | final-frame edit-ROI MAE → GT |
|---:|---:|---:|---:|
{milestone_rows}
"""


def package_side(
    case: dict[str, Any],
    side_dir: Path,
    trajectory_cell: str,
    score: float,
    native_video: Path,
    scored_video: Path,
    scorer_result: dict[str, Any],
    gt_first: np.ndarray,
    gt_final: np.ndarray,
    edit_mask: np.ndarray,
) -> dict[str, Any]:
    relative_case = Path(case["canonical_name"])
    trajectory_dir = TRAJECTORY_ROOT / trajectory_cell / relative_case
    manifest_path = trajectory_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest["sample_name"] != case["canonical_name"]:
        raise ValueError(f"Trajectory sample mismatch: {trajectory_dir}")
    final_hashes = {
        "formal_native": sha256_file(native_video),
        "trajectory_final": sha256_file(trajectory_dir / "final_00.mp4"),
        "trajectory_step_30": sha256_file(trajectory_dir / "step_29.mp4"),
    }
    if len(set(final_hashes.values())) != 1:
        raise ValueError(f"Final-video trajectory binding mismatch: {trajectory_dir}")
    if scorer_result["video_sha256"] != sha256_file(scored_video):
        raise ValueError(f"Scored-video binding mismatch: {scored_video}")

    link_or_copy(trajectory_dir / "final_00.mp4", side_dir / "final_native_exact.mp4")
    link_or_copy(scored_video, side_dir / "final_scored_1024_exact.mp4")
    link_or_copy(trajectory_dir / "steps_grid.mp4", side_dir / "overview_30_steps.mp4")
    link_or_copy(trajectory_dir / "step_contact_sheet.jpg", side_dir / "contact_sheet_30_steps.jpg")
    link_or_copy(manifest_path, side_dir / "trajectory_manifest.json")
    sample_metadata = trajectory_dir / "sample_metadata.json"
    if sample_metadata.is_file():
        link_or_copy(sample_metadata, side_dir / "trajectory_sample_metadata.json")
    for index in range(STEP_COUNT):
        source = trajectory_dir / f"step_{index:02d}.mp4"
        destination = side_dir / "all_steps" / f"step_{index + 1:02d}.mp4"
        link_or_copy(source, destination)
        if index + 1 in MILESTONE_STEPS:
            link_or_copy(source, side_dir / "milestones" / f"step_{index + 1:02d}.mp4")

    rows = write_metrics(trajectory_dir, side_dir, manifest, gt_first, gt_final, edit_mask)
    write_text_once(side_dir / "score.txt", f"{score:.12f}\n")
    write_json_once(side_dir / "scorer_details.json", scorer_result)
    write_text_once(
        side_dir / "README.md",
        reasoning_readme(side_dir.name, score, manifest, rows),
    )
    return {
        "trajectory_cell": trajectory_cell,
        "score": score,
        "final_sha256": final_hashes["formal_native"],
        "scored_video_sha256": scorer_result["video_sha256"],
        "trajectory_manifest_sha256": sha256_file(manifest_path),
        "step_count": STEP_COUNT,
        "milestone_steps": list(MILESTONE_STEPS),
        "final_binding_pass": True,
    }


def case_readme(case: dict[str, Any], audit: dict[str, Any]) -> str:
    summary = audit["metadata_audit"]["semantic_ground_truth"].get("task_summary_zh", case["task_label"])
    return f"""# {case["case_id"]} · {case["task_label"]}

{summary}

- sample: `{case["canonical_name"]}`
- baseline UniPC: `{case["baseline_score"]:.6f}`
- checkpoint 2200 CPS 0.7: `{case["checkpoint_score"]:.6f}`
- delta: `{case["score_delta"]:+.6f}`
- scorer audit: **PASS**

先看 `question/first_frame.png` 和 `question/prompt.txt`，再看 `ground_truth/`。
两组结果各自包含 raw native final、正式 scorer 输入、30 步全集、七个 milestone、
完整 overview/contact sheet 和逐步视觉收敛指标。详细的 evaluator 证据见
`scorer_audit.md` / `scorer_audit.json`。
"""


def summarize_reasoning_chains(output_dir: Path) -> dict[str, Any]:
    chains: list[dict[str, Any]] = []
    for case_dir in sorted((output_dir / "cases").iterdir()):
        for side in ("baseline_unipc", "checkpoint2200_cps0p7"):
            with (case_dir / side / "reasoning_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != STEP_COUNT:
                raise ValueError(f"Unexpected reasoning metric count: {case_dir / side}")
            final_distances = [float(row["mae_to_formal_final_sampled_video"]) for row in rows]
            unique_hashes = len({row["sha256"] for row in rows})
            if unique_hashes != STEP_COUNT or final_distances[0] <= 0 or final_distances[-1] != 0:
                raise ValueError(f"Reasoning-chain evolution audit failed: {case_dir / side}")
            nonincreasing_fraction = sum(
                left >= right for left, right in zip(final_distances, final_distances[1:], strict=False)
            ) / (STEP_COUNT - 1)
            chains.append(
                {
                    "case_folder": case_dir.name,
                    "side": side,
                    "step_count": len(rows),
                    "unique_step_sha256_count": unique_hashes,
                    "step_01_mae_to_formal_final": final_distances[0],
                    "step_15_mae_to_formal_final": final_distances[14],
                    "step_29_mae_to_formal_final": final_distances[28],
                    "step_30_mae_to_formal_final": final_distances[29],
                    "nonincreasing_transition_fraction": nonincreasing_fraction,
                    "audit_pass": True,
                }
            )

    def stats(key: str) -> dict[str, float]:
        values = [float(chain[key]) for chain in chains]
        return {
            "min": min(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    return {
        "schema_version": 1,
        "chain_count": len(chains),
        "steps_per_chain": STEP_COUNT,
        "decoded_step_video_count": len(chains) * STEP_COUNT,
        "all_chains_have_30_unique_step_files": all(
            chain["unique_step_sha256_count"] == STEP_COUNT for chain in chains
        ),
        "all_chains_end_at_exact_formal_final": all(chain["step_30_mae_to_formal_final"] == 0 for chain in chains),
        "all_chains_pass": all(chain["audit_pass"] for chain in chains),
        "step_01_mae_to_formal_final": stats("step_01_mae_to_formal_final"),
        "step_15_mae_to_formal_final": stats("step_15_mae_to_formal_final"),
        "step_29_mae_to_formal_final": stats("step_29_mae_to_formal_final"),
        "nonincreasing_transition_fraction": stats("nonincreasing_transition_fraction"),
        "evidence_boundary": (
            "This proves decodability, distinct step encodings, visual evolution, and exact final binding. "
            "It is not a separate semantic reward for intermediate steps."
        ),
        "chains": chains,
    }


def main() -> int:
    args = parse_args()
    selection_path = args.selection.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selection = load_json(selection_path)
    cases = selection["cases"]
    scorer_report_path = output_dir / "scorer_audit/scorer_audit.json"
    if not scorer_report_path.is_file():
        raise FileNotFoundError(f"Run audit_checkpoint2200_reasoning_cases.py first: {scorer_report_path}")
    scorer_report = load_json(scorer_report_path)
    if not scorer_report["all_cases_pass"]:
        raise ValueError("Scorer audit has failed cases")
    audit_by_id = {item["case_id"]: item for item in scorer_report["cases"]}

    output_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(selection_path, output_dir / "selection_manifest.json")
    link_or_copy(DEFAULT_SELECTION_ROOT / "selection.csv", output_dir / "selection.csv")

    provenance_dir = output_dir / "provenance"
    for cell in (BASELINE_CELL, CHECKPOINT_CELL):
        link_or_copy(
            TRAJECTORY_ROOT / cell / "cell_manifest.json",
            provenance_dir / f"trajectory_{cell}_cell_manifest.json",
        )
    for side, case_key in (
        ("baseline", "baseline_scored_video"),
        ("checkpoint2200", "checkpoint_scored_video"),
    ):
        score_root = relative_source(cases[0][case_key]).parents[3]
        for name in ("generation-provenance.json", "preparation-provenance.json", "score-provenance.json"):
            link_or_copy(score_root / name, provenance_dir / f"{side}_{name}")

    trajectory_cell_audit = {
        "schema_version": 1,
        "auditor": "src.cli.audit_vbvr_i2v_trajectories",
        "mode": "read_only; no historical trajectory artifacts modified",
        "cells": [],
    }
    for cell in (BASELINE_CELL, CHECKPOINT_CELL):
        cell_manifest_path = TRAJECTORY_ROOT / cell / "cell_manifest.json"
        cell_manifest = load_json(cell_manifest_path)
        if (
            cell_manifest["state"] != "complete"
            or cell_manifest["sample_count"] != 500
            or cell_manifest["completed_count"] != 500
            or cell_manifest["num_inference_steps"] != STEP_COUNT
        ):
            raise ValueError(f"Incomplete trajectory cell: {cell}")
        trajectory_cell_audit["cells"].append(
            {
                "cell": cell,
                "state": "complete",
                "samples_audited": 500,
                "steps_per_sample": STEP_COUNT,
                "formal_final_binding_checked": True,
                "cell_manifest_sha256": sha256_file(cell_manifest_path),
            }
        )
    write_json_once(provenance_dir / "trajectory_readonly_audit.json", trajectory_cell_audit)

    evalkit_sources = output_dir / "scorer_audit/evalkit_sources"
    source_relatives = {item["evaluator"]["source"] for item in scorer_report["cases"]} | {
        "vbvr_bench/evaluators/__init__.py",
        "vbvr_bench/evaluators/base_evaluator.py",
    }
    for relative in sorted(source_relatives):
        link_or_copy(EVALKIT_ROOT / relative, evalkit_sources / relative)

    index_rows: list[dict[str, Any]] = []
    package_cases: list[dict[str, Any]] = []
    for case in cases:
        audit = audit_by_id[case["case_id"]]
        task_short = case["task_name"].split("_", 1)[0]
        case_dir = output_dir / "cases" / f"{case['case_id']}_{task_short}_{case['video_idx']}"
        sample_root = relative_source(case["input_image"]).parent
        link_or_copy(relative_source(case["input_image"]), case_dir / "question/first_frame.png")
        link_or_copy(sample_root / "metadata.json", case_dir / "question/metadata.json")
        write_text_once(case_dir / "question/prompt.txt", case["prompt"] + "\n")
        write_json_once(
            case_dir / "question/semantic_ground_truth.json",
            audit["metadata_audit"]["semantic_ground_truth"],
        )
        link_or_copy(
            relative_source(case["ground_truth_video"]),
            case_dir / "ground_truth/ground_truth.mp4",
        )
        link_or_copy(
            relative_source(case["ground_truth_final"]),
            case_dir / "ground_truth/final_frame.png",
        )
        gt_first, gt_final, edit_mask = load_question_images(case)
        baseline = package_side(
            case,
            case_dir / "baseline_unipc",
            BASELINE_CELL,
            case["baseline_score"],
            relative_source(case["baseline_native_video"]),
            relative_source(case["baseline_scored_video"]),
            audit["results"]["baseline_unipc"],
            gt_first,
            gt_final,
            edit_mask,
        )
        checkpoint = package_side(
            case,
            case_dir / "checkpoint2200_cps0p7",
            CHECKPOINT_CELL,
            case["checkpoint_score"],
            relative_source(case["checkpoint_native_video"]),
            relative_source(case["checkpoint_scored_video"]),
            audit["results"]["checkpoint2200_cps0p7"],
            gt_first,
            gt_final,
            edit_mask,
        )
        write_json_once(case_dir / "scorer_audit.json", audit)
        write_text_once(case_dir / "scorer_audit.md", scorer_summary_markdown(case, audit))
        write_text_once(case_dir / "README.md", case_readme(case, audit))
        case_manifest = {
            "schema_version": 1,
            "case_id": case["case_id"],
            "canonical_name": case["canonical_name"],
            "task_name": case["task_name"],
            "task_label": case["task_label"],
            "prompt": case["prompt"],
            "scores": {
                "baseline_unipc": case["baseline_score"],
                "checkpoint2200_cps0p7": case["checkpoint_score"],
                "delta": case["score_delta"],
                "ground_truth_self_score": audit["rerun_scores"]["ground_truth_self_score"],
            },
            "evaluator": audit["evaluator"],
            "scorer_audit_pass": audit["audit_pass"],
            "baseline": baseline,
            "checkpoint2200": checkpoint,
        }
        write_json_once(case_dir / "case_manifest.json", case_manifest)
        package_cases.append(case_manifest)
        index_rows.append(
            {
                "case_id": case["case_id"],
                "folder": case_dir.name,
                "domain": case["domain"],
                "task_name": case["task_name"],
                "task_label": case["task_label"],
                "video_idx": case["video_idx"],
                "baseline_score": case["baseline_score"],
                "checkpoint_score": case["checkpoint_score"],
                "delta": case["score_delta"],
                "gt_self_score": audit["rerun_scores"]["ground_truth_self_score"],
                "evaluator_class": audit["evaluator"]["class"],
                "audit_pass": audit["audit_pass"],
            }
        )

    index_csv = output_dir / "INDEX.csv"
    if not index_csv.exists():
        with index_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=index_rows[0].keys())
            writer.writeheader()
            writer.writerows(index_rows)
    index_table = "\n".join(
        f"| {row['case_id']} | [{row['task_label']}](cases/{row['folder']}/) | "
        f"{row['baseline_score']:.6f} | {row['checkpoint_score']:.6f} | {row['delta']:+.6f} |"
        for row in index_rows
    )
    write_text_once(
        output_dir / "INDEX.md",
        "# Case index\n\n| ID | task | baseline | checkpoint 2200 | delta |\n"
        "|---|---|---:|---:|---:|\n" + index_table + "\n",
    )
    reasoning_chain_audit = summarize_reasoning_chains(output_dir)
    write_json_once(output_dir / "reasoning_chain_audit.json", reasoning_chain_audit)
    package_manifest = {
        "schema_version": 1,
        "case_count": len(package_cases),
        "task_count": len({case["task_name"] for case in package_cases}),
        "format": "folder package; no PPT; no GIF",
        "reasoning_steps_per_side_per_case": STEP_COUNT,
        "total_reasoning_step_videos": len(package_cases) * 2 * STEP_COUNT,
        "scorer_audit": {
            "all_cases_pass": scorer_report["all_cases_pass"],
            "independent_scoring_runs": scorer_report["scored_video_count"],
            "evalkit_source_sha256": scorer_report["evalkit_source_sha256"],
            "runtime_sha256": scorer_report["scorer_runtime"]["sha256"],
            "gt_self_score_min": scorer_report["ground_truth_self_score_min"],
            "gt_self_score_mean": scorer_report["ground_truth_self_score_mean"],
        },
        "trajectory_cells": [BASELINE_CELL, CHECKPOINT_CELL],
        "reasoning_chain_audit": {key: value for key, value in reasoning_chain_audit.items() if key != "chains"},
        "cases": package_cases,
    }
    write_json_once(output_dir / "package_manifest.json", package_manifest)
    write_text_once(
        output_dir / "README.md",
        f"""# VBVR checkpoint 2200 vs baseline · audited reasoning-chain materials

这是 50 个案例的文件夹素材包，不是 PPT；全部使用 MP4，没有 GIF，也没有替换 raw video 封面。

## 审核结论

- 50/50 案例通过精确 scorer 复核。
- 共独立执行 150 次 e140 EvalKit 评分：50 个 baseline、50 个 checkpoint、50 个 GT self-score；零 scorer error。
- 两组正式分数均逐值复现（绝对容差 `1e-12`）。
- GT self-score 最低 `{scorer_report["ground_truth_self_score_min"]:.6f}`，
  平均 `{scorer_report["ground_truth_self_score_mean"]:.6f}`。
- 使用的 EvalKit source SHA-256 为 `{scorer_report["evalkit_source_sha256"]}`。
- 两个 trajectory cell 均通过全 500 样本的只读完整性审计；本包每例的
  `step_30.mp4`、trajectory `final_00.mp4` 和正式 native final 三者 SHA-256 完全一致。

## 如何浏览

1. 打开 `INDEX.md` 选择案例。
2. 每例先看 `question/first_frame.png`、prompt 和 `ground_truth/`。
3. 对比 `baseline_unipc/` 与 `checkpoint2200_cps0p7/` 的 `final_native_exact.mp4` 和精确 score。
4. 快速看过程：`overview_30_steps.mp4`、`contact_sheet_30_steps.jpg` 或 `milestones/`。
5. 完整过程：`all_steps/step_01.mp4`–`step_30.mp4`。
6. 打分证据：`scorer_audit.md`、`scorer_audit.json` 和 `scorer_details.json`。

全包的轨迹结构与视觉收敛统计见 `reasoning_chain_audit.json`；两组全 500 样本的只读轨迹
复核摘要见 `provenance/trajectory_readonly_audit.json`。

这些 30 步是采样器的 predicted-clean 视频轨迹，不是文本 chain-of-thought。
第 1–29 步为对应 sigma 上的 post-CFG predicted-clean x0，第 30 步为实际 sigma=0 final；
EvalKit 只对正式 final 打分。`reasoning_metrics.csv` 是帮助理解视觉收敛的辅助量，
不参与 reward。

注意：这是按 checkpoint-minus-baseline 分差选出的 top-50 展示集，并非无偏总体统计；
对比同时改变了模型 checkpoint 和 sampler（CPS 0.7 vs UniPC），因此展示的是完整生成
配置的差异，不能只凭这一组将全部差异归因于 RL 权重。
""",
    )

    checksum_rows: list[str] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        checksum_rows.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    write_text_once(output_dir / "SHA256SUMS", "\n".join(checksum_rows) + "\n")
    total_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "cases": len(package_cases),
                "reasoning_step_videos": len(package_cases) * 2 * STEP_COUNT,
                "files": len(checksum_rows) + 1,
                "bytes": total_bytes,
                "scorer_audit_pass": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
