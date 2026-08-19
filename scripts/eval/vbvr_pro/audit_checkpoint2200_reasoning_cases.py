#!/usr/bin/env python3
"""Re-score and audit the selected checkpoint-2200 reasoning-chain cases.

The audit deliberately uses the exact prepared 1024x1024 videos and pinned
EvalKit checkout that produced the published scores.  In addition to checking
that the scores reproduce, it records the concrete evaluator selected by the
EvalKit registry, the task metadata contract, the evaluator's task-specific
details, and a ground-truth self-score for every selected sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.eval import vbvr_run_evaluation_parallel as parallel_eval
from src.eval.vbvr_runtime import validate_vbvr_scorer_runtime

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTION = REPO_ROOT / "storage/presentations/vbvr_checkpoint2200_vs_baseline_20260818/selection_manifest.json"
DEFAULT_EVALKIT = REPO_ROOT / "storage/evalkits/vbvr-evalkit-interleave-main_v2-e140038f"
DEFAULT_GT_ROOT = REPO_ROOT / "storage/datasets/vbvr-pro-eval-500"
DEFAULT_OUTPUT = (
    REPO_ROOT / "storage/presentations/vbvr_checkpoint2200_vs_baseline_reasoning_chains_20260818/scorer_audit"
)
SCORE_ABS_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--evalkit-dir", type=Path, default=DEFAULT_EVALKIT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    return repr(value)


def metadata_path_exists(metadata: dict[str, Any], dotted_path: str) -> bool:
    parts = dotted_path.split(".")
    if parts[0] in {"semantic_ground_truth", "parameters"}:
        roots = [metadata]
    else:
        semantic = metadata.get("semantic_ground_truth", {})
        roots = [
            semantic,
            semantic.get("interpretation", {}) if isinstance(semantic, Mapping) else {},
            metadata.get("parameters", {}),
        ]
    for root in roots:
        value: Any = root
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            return True
    return False


def _init_worker(evalkit_dir: str, threads_per_worker: int) -> None:
    parallel_eval._init_worker(  # noqa: SLF001 - exact shared scorer initializer
        evalkit_dir,
        threads_per_worker,
        easyocr_module_path=os.environ.get("EASYOCR_MODULE_PATH"),
        hide_cuda=True,
    )


def _score_one(task: dict[str, Any]) -> dict[str, Any]:
    rek = parallel_eval._WORKER_REK  # noqa: SLF001 - initialized above
    if rek is None:
        raise RuntimeError("EvalKit worker was not initialized")

    gt_info = rek.find_gt_info(task["task_name"], task["video_idx"], task["gt_root"])
    evaluator = rek.get_evaluator(task["task_name"], "cpu")
    eval_info = {
        "video_path": task["video_path"],
        "task_name": task["task_name"],
        "no_ssim_fallback": True,
        **gt_info,
    }
    result = evaluator.evaluate(eval_info, task_specific_only=True)
    normalized = parallel_eval._normalize_evalkit_result(result)  # noqa: SLF001
    evaluator_source = Path(inspect.getsourcefile(evaluator.__class__) or "").resolve()
    evalkit_dir = Path(task["evalkit_dir"]).resolve()
    try:
        evaluator_source_display = evaluator_source.relative_to(evalkit_dir).as_posix()
    except ValueError:
        evaluator_source_display = str(evaluator_source)
    return {
        "case_id": task["case_id"],
        "rank": task["rank"],
        "canonical_name": task["canonical_name"],
        "task_name": task["task_name"],
        "side": task["side"],
        "video_path": task["video_path"],
        "video_sha256": sha256_file(Path(task["video_path"])),
        "score": normalized["score"],
        "dimensions": jsonable(normalized["dimensions"]),
        "error": normalized["error"],
        "result_details": jsonable(result.get("details", {})),
        "task_specific_details": jsonable(getattr(evaluator, "_last_task_details", {})),
        "evaluator_class": evaluator.__class__.__name__,
        "evaluator_source": evaluator_source_display,
        "evaluator_source_line": inspect.getsourcelines(evaluator.__class__)[1],
    }


def build_tasks(
    cases: list[dict[str, Any]], evalkit_dir: Path, gt_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    metadata_audits: dict[str, dict[str, Any]] = {}
    for case in cases:
        first_frame = (REPO_ROOT / case["input_image"]).resolve()
        sample_root = first_frame.parent
        metadata_path = sample_root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scoring_contract = metadata.get("scoring_contract", {})
        required_fields = scoring_contract.get("required_semantic_fields", [])
        missing_fields = [field for field in required_fields if not metadata_path_exists(metadata, field)]
        metadata_audits[case["case_id"]] = {
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "evalkit_map_key": scoring_contract.get("evalkit_map_key"),
            "declared_evaluator_class": scoring_contract.get("evaluator_class"),
            "required_semantic_fields": required_fields,
            "missing_required_semantic_fields": missing_fields,
            "semantic_ground_truth": metadata.get("semantic_ground_truth", {}),
            "metadata_contract_pass": (
                scoring_contract.get("evalkit_map_key") == case["task_name"] and not missing_fields
            ),
        }
        side_paths = {
            "baseline_unipc": (REPO_ROOT / case["baseline_scored_video"]).resolve(),
            "checkpoint2200_cps0p7": (REPO_ROOT / case["checkpoint_scored_video"]).resolve(),
            "ground_truth": sample_root / "ground_truth.mp4",
        }
        for side, video_path in side_paths.items():
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            tasks.append(
                {
                    "case_id": case["case_id"],
                    "rank": case["rank"],
                    "canonical_name": case["canonical_name"],
                    "task_name": case["task_name"],
                    "video_idx": int(case["video_idx"]),
                    "side": side,
                    "video_path": str(video_path),
                    "gt_root": str(gt_root),
                    "evalkit_dir": str(evalkit_dir),
                }
            )
    return tasks, metadata_audits


def main() -> int:
    args = parse_args()
    if args.num_workers < 1 or args.threads_per_worker < 1:
        raise ValueError("Worker and thread counts must be positive")

    selection_path = args.selection.expanduser().resolve()
    evalkit_dir = args.evalkit_dir.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cases = selection["cases"]
    if len(cases) != selection["selected_case_count"]:
        raise ValueError("Selection case count does not match its manifest")

    runtime_report = validate_vbvr_scorer_runtime()
    evalkit_sha256 = parallel_eval.evalkit_source_sha256(evalkit_dir)
    expected_evalkit_sha256 = selection["source_audit"]["evalkit_source_sha256"]
    if evalkit_sha256 != expected_evalkit_sha256:
        raise ValueError(f"EvalKit fingerprint mismatch: expected={expected_evalkit_sha256}, actual={evalkit_sha256}")

    tasks, metadata_audits = build_tasks(cases, evalkit_dir, gt_root)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    os.environ["EASYOCR_MODULE_PATH"] = str((REPO_ROOT / "storage/evalkits/easyocr-shared").resolve())
    with Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(str(evalkit_dir), args.threads_per_worker),
    ) as pool:
        reruns = list(
            tqdm(
                pool.imap(_score_one, tasks),
                total=len(tasks),
                desc="Re-scoring selected cases and GT",
            )
        )

    rerun_by_key = {(item["case_id"], item["side"]): item for item in reruns}
    case_audits: list[dict[str, Any]] = []
    for case in cases:
        baseline = rerun_by_key[(case["case_id"], "baseline_unipc")]
        checkpoint = rerun_by_key[(case["case_id"], "checkpoint2200_cps0p7")]
        ground_truth = rerun_by_key[(case["case_id"], "ground_truth")]
        score_checks = {
            "baseline_reproduced": math.isclose(
                baseline["score"], case["baseline_score"], rel_tol=0.0, abs_tol=SCORE_ABS_TOLERANCE
            ),
            "checkpoint_reproduced": math.isclose(
                checkpoint["score"],
                case["checkpoint_score"],
                rel_tol=0.0,
                abs_tol=SCORE_ABS_TOLERANCE,
            ),
            "score_order_reproduced": checkpoint["score"] > baseline["score"],
            "all_errors_null": all(item["error"] is None for item in (baseline, checkpoint, ground_truth)),
        }
        evaluator_checks = {
            "same_evaluator_for_all_three": len(
                {
                    (item["evaluator_class"], item["evaluator_source"], item["evaluator_source_line"])
                    for item in (baseline, checkpoint, ground_truth)
                }
            )
            == 1,
            "registry_key_matches_metadata": metadata_audits[case["case_id"]]["evalkit_map_key"] == case["task_name"],
            "metadata_contract_pass": metadata_audits[case["case_id"]]["metadata_contract_pass"],
        }
        audit_pass = all(score_checks.values()) and all(evaluator_checks.values())
        case_audits.append(
            {
                "case_id": case["case_id"],
                "rank": case["rank"],
                "canonical_name": case["canonical_name"],
                "task_name": case["task_name"],
                "recorded_scores": {
                    "baseline_unipc": case["baseline_score"],
                    "checkpoint2200_cps0p7": case["checkpoint_score"],
                    "delta": case["score_delta"],
                },
                "rerun_scores": {
                    "baseline_unipc": baseline["score"],
                    "checkpoint2200_cps0p7": checkpoint["score"],
                    "ground_truth_self_score": ground_truth["score"],
                    "delta": checkpoint["score"] - baseline["score"],
                },
                "score_checks": score_checks,
                "evaluator_checks": evaluator_checks,
                "evaluator": {
                    "class": baseline["evaluator_class"],
                    "source": baseline["evaluator_source"],
                    "source_line": baseline["evaluator_source_line"],
                    "declared_metadata_class": metadata_audits[case["case_id"]]["declared_evaluator_class"],
                },
                "metadata_audit": metadata_audits[case["case_id"]],
                "results": {
                    "baseline_unipc": baseline,
                    "checkpoint2200_cps0p7": checkpoint,
                    "ground_truth": ground_truth,
                },
                "audit_pass": audit_pass,
            }
        )

    failed = [item["case_id"] for item in case_audits if not item["audit_pass"]]
    gt_scores = [item["rerun_scores"]["ground_truth_self_score"] for item in case_audits]
    report = {
        "schema_version": 1,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "evalkit_dir": str(evalkit_dir),
        "evalkit_source_sha256": evalkit_sha256,
        "scorer_runtime": runtime_report,
        "gt_root": str(gt_root),
        "case_count": len(case_audits),
        "scored_video_count": len(reruns),
        "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        "all_cases_pass": not failed,
        "failed_case_ids": failed,
        "ground_truth_self_score_min": min(gt_scores),
        "ground_truth_self_score_mean": sum(gt_scores) / len(gt_scores),
        "cases": case_audits,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "scorer_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "scorer_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "canonical_name",
                "evaluator_class",
                "evaluator_source",
                "evaluator_source_line",
                "baseline_recorded",
                "baseline_rerun",
                "checkpoint_recorded",
                "checkpoint_rerun",
                "ground_truth_self_score",
                "delta_rerun",
                "audit_pass",
            ),
        )
        writer.writeheader()
        for item in case_audits:
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "canonical_name": item["canonical_name"],
                    "evaluator_class": item["evaluator"]["class"],
                    "evaluator_source": item["evaluator"]["source"],
                    "evaluator_source_line": item["evaluator"]["source_line"],
                    "baseline_recorded": item["recorded_scores"]["baseline_unipc"],
                    "baseline_rerun": item["rerun_scores"]["baseline_unipc"],
                    "checkpoint_recorded": item["recorded_scores"]["checkpoint2200_cps0p7"],
                    "checkpoint_rerun": item["rerun_scores"]["checkpoint2200_cps0p7"],
                    "ground_truth_self_score": item["rerun_scores"]["ground_truth_self_score"],
                    "delta_rerun": item["rerun_scores"]["delta"],
                    "audit_pass": item["audit_pass"],
                }
            )

    print(json.dumps({key: report[key] for key in report if key != "cases"}, indent=2))
    return 0 if report["all_cases_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
