from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from src.eval.evaluation_provenance import verify_recorded_manifest

RESULT_RELATIVE_PATH = Path("scores/eval_1024x1024_161f_5s_vbvr_results.json")
SCORE_PROVENANCE_NAME = "score-provenance.json"
DANCE_RUN_RE = re.compile(r"^dancegrpo_vbvr_pro_5b_checkpoint-(?P<step>\d+)(?:-cps-noise-(?P<noise>\d+(?:\.\d+)?))?$")
CATEGORY_ORDER = ("Abstraction", "Spatiality", "Transformation", "Perception", "Knowledge")


@dataclass(frozen=True)
class RunResult:
    name: str
    family: str
    step: int | None
    sampler: str
    noise_level: float | None
    result_path: Path
    summary: dict[str, Any]
    task_scores: dict[str, float]
    task_meta: dict[str, tuple[str, str]]
    task_counts: dict[str, int]
    evalkit_revision: str
    evalkit_source_sha256: str

    @property
    def overall(self) -> float:
        return float(self.summary["overall"]["mean_score"])

    @property
    def in_domain(self) -> float:
        return float(self.summary["In_Domain"]["mean_score"])

    @property
    def out_of_domain(self) -> float:
        return float(self.summary["Out_of_Domain"]["mean_score"])

    @property
    def sample_count(self) -> int:
        return int(self.summary["overall"]["num_samples"])


def _classify_run(name: str) -> tuple[str, int | None, str, float | None]:
    match = DANCE_RUN_RE.fullmatch(name)
    if match:
        noise = match.group("noise")
        return (
            "DanceGRPO",
            int(match.group("step")),
            "CPS" if noise is not None else "ODE",
            (float(noise) if noise is not None else None),
        )
    if name.startswith("sft_"):
        return "SFT", None, "ODE", None
    return "Other", None, "Unknown", None


def _load_run(
    run_dir: Path,
    expected_samples: int,
    expected_evalkit_source_sha256: str | None = None,
) -> RunResult:
    result_path = run_dir / RESULT_RELATIVE_PATH
    provenance_path = run_dir / SCORE_PROVENANCE_NAME
    if not provenance_path.is_file():
        raise ValueError(f"{run_dir.name}: missing {SCORE_PROVENANCE_NAME}")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("schema_version") != 1:
        raise ValueError(f"{run_dir.name}: unsupported or missing score provenance schema")
    if provenance.get("stage") != "vbvr-pro-score":
        raise ValueError(f"{run_dir.name}: score provenance stage is not vbvr-pro-score")
    values = provenance.get("values", {})
    if values.get("state") != "complete":
        raise ValueError(f"{run_dir.name}: score provenance is not complete")
    evalkit_revision = values.get("evalkit_revision")
    evalkit_source_sha256 = values.get("evalkit_source_sha256")
    if not isinstance(evalkit_revision, str) or not evalkit_revision:
        raise ValueError(f"{run_dir.name}: score provenance is missing evalkit_revision")
    if not isinstance(evalkit_source_sha256, str) or len(evalkit_source_sha256) != 64:
        raise ValueError(f"{run_dir.name}: score provenance is missing evalkit_source_sha256")
    if expected_evalkit_source_sha256 and evalkit_source_sha256 != expected_evalkit_source_sha256:
        raise ValueError(
            f"{run_dir.name}: scorer fingerprint {evalkit_source_sha256} "
            f"does not match expected {expected_evalkit_source_sha256}"
        )
    recorded_result = provenance.get("output_files", {}).get("result", {})
    if recorded_result.get("path") != str(result_path.resolve()):
        raise ValueError(f"{run_dir.name}: score provenance is not bound to its standard result JSON")
    matches, detail = verify_recorded_manifest(
        provenance_path,
        expected_stage="vbvr-pro-score",
        require_complete=True,
        sections=("output_files",),
    )
    if not matches:
        raise ValueError(f"{run_dir.name}: {detail}")
    result = json.loads(result_path.read_text())
    samples = result.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        raise ValueError(f"{run_dir.name}: expected {expected_samples} samples, found {len(samples or [])}")
    errors = [sample for sample in samples if sample.get("error")]
    if errors:
        raise ValueError(f"{run_dir.name}: scorer errors={len(errors)}")

    task_meta: dict[str, tuple[str, str]] = {}
    task_counts: Counter[str] = Counter()
    for sample in samples:
        task_name = str(sample["task_name"])
        metadata = (str(sample["split"]), str(sample["category"]))
        previous = task_meta.setdefault(task_name, metadata)
        if previous != metadata:
            raise ValueError(f"{run_dir.name}: inconsistent metadata for task {task_name}")
        task_counts[task_name] += 1

    summary = result["summary"]
    task_scores = {name: float(score) for name, score in summary["overall"]["by_task"].items()}
    if task_scores.keys() != task_meta.keys():
        raise ValueError(f"{run_dir.name}: task summary does not match sample task set")
    family, step, sampler, noise_level = _classify_run(run_dir.name)
    return RunResult(
        name=run_dir.name,
        family=family,
        step=step,
        sampler=sampler,
        noise_level=noise_level,
        result_path=result_path,
        summary=summary,
        task_scores=task_scores,
        task_meta=task_meta,
        task_counts=dict(task_counts),
        evalkit_revision=evalkit_revision,
        evalkit_source_sha256=evalkit_source_sha256,
    )


def discover_runs(
    root: Path,
    expected_samples: int = 500,
    expected_evalkit_source_sha256: str | None = None,
) -> tuple[list[RunResult], list[tuple[str, str]]]:
    runs: list[RunResult] = []
    skipped: list[tuple[str, str]] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        result_path = run_dir / RESULT_RELATIVE_PATH
        if not result_path.is_file():
            if DANCE_RUN_RE.fullmatch(run_dir.name):
                skipped.append((run_dir.name, "missing standard result JSON"))
            continue
        try:
            runs.append(_load_run(run_dir, expected_samples, expected_evalkit_source_sha256))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            skipped.append((run_dir.name, str(exc)))

    family_order = {"SFT": 0, "DanceGRPO": 1, "Other": 2}
    sampler_order = {"ODE": 0, "CPS": 1, "Unknown": 2}
    runs.sort(
        key=lambda run: (
            family_order.get(run.family, 9),
            run.step if run.step is not None else -1,
            sampler_order.get(run.sampler, 9),
            run.noise_level if run.noise_level is not None else -1.0,
            run.name,
        )
    )
    scorer_hashes = {run.evalkit_source_sha256 for run in runs}
    if len(scorer_hashes) > 1:
        raise ValueError(
            "Refusing to mix VBVR-Pro results from multiple scorer fingerprints: " + ", ".join(sorted(scorer_hashes))
        )
    return runs, skipped


def _format_sheet(sheet, widths: list[int], score_columns: list[int] | None = None) -> None:
    header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for column in score_columns or []:
        for cell in sheet.iter_cols(min_col=column, max_col=column, min_row=2):
            for item in cell:
                item.number_format = "0.000000"


def _category_score(run: RunResult, category: str) -> float | None:
    value = run.summary["overall"].get("by_category", {}).get(category)
    return float(value) if value is not None else None


def write_all_run_summary(runs: list[RunResult], skipped: list[tuple[str, str]], output_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "All Runs"
    sheet.append(
        [
            "Run",
            "Family",
            "Checkpoint Step",
            "Sampler",
            "CPS Noise Level",
            "Overall",
            "In-Domain",
            "Out-of-Domain",
            *CATEGORY_ORDER,
            "Samples",
            "Tasks",
            "EvalKit Revision",
            "EvalKit Source SHA-256",
            "Result JSON",
        ]
    )
    for run in runs:
        sheet.append(
            [
                run.name,
                run.family,
                run.step,
                run.sampler,
                run.noise_level,
                run.overall,
                run.in_domain,
                run.out_of_domain,
                *(_category_score(run, category) for category in CATEGORY_ORDER),
                run.sample_count,
                len(run.task_scores),
                run.evalkit_revision,
                run.evalkit_source_sha256,
                str(run.result_path),
            ]
        )
    _format_sheet(
        sheet,
        [72, 16, 18, 12, 18, 14, 14, 16, 16, 16, 18, 16, 16, 12, 10, 42, 68, 90],
        list(range(6, 14)),
    )

    skipped_sheet = workbook.create_sheet("Skipped")
    skipped_sheet.append(["Run", "Reason"])
    for name, reason in skipped:
        skipped_sheet.append([name, reason])
    _format_sheet(skipped_sheet, [72, 72])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def _task_sort_key(task: str, metadata: dict[str, tuple[str, str]]) -> tuple[int, str]:
    split = metadata[task][0]
    return ({"In_Domain": 0, "Out_of_Domain": 1}.get(split, 2), task)


def write_cps_task_changes(runs: list[RunResult], skipped: list[tuple[str, str]], output_path: Path) -> None:
    ode_by_step = {run.step: run for run in runs if run.family == "DanceGRPO" and run.sampler == "ODE"}
    cps_runs = [run for run in runs if run.family == "DanceGRPO" and run.sampler == "CPS"]

    workbook = Workbook()
    long_sheet = workbook.active
    long_sheet.title = "All Deltas"
    long_sheet.append(
        ["Checkpoint Step", "CPS Noise Level", "Split", "Category", "Task Name", "Samples", "ODE", "CPS", "Delta"]
    )
    for cps in sorted(cps_runs, key=lambda run: (run.step or -1, run.noise_level or -1.0)):
        ode = ode_by_step.get(cps.step)
        if ode is None:
            continue
        for task in sorted(cps.task_scores, key=lambda name: _task_sort_key(name, cps.task_meta)):
            split, category = cps.task_meta[task]
            ode_score = ode.task_scores[task]
            cps_score = cps.task_scores[task]
            long_sheet.append(
                [
                    cps.step,
                    cps.noise_level,
                    split.replace("_", "-"),
                    category,
                    task,
                    cps.task_counts[task],
                    ode_score,
                    cps_score,
                    cps_score - ode_score,
                ]
            )
    _format_sheet(long_sheet, [18, 18, 18, 18, 78, 12, 14, 14, 14], [7, 8, 9])

    noise_levels = sorted({run.noise_level for run in cps_runs if run.noise_level is not None})
    for noise_level in noise_levels:
        level_runs = sorted((run for run in cps_runs if run.noise_level == noise_level), key=lambda run: run.step or -1)
        comparable = [(ode_by_step[run.step], run) for run in level_runs if run.step in ode_by_step]
        if not comparable:
            continue
        sheet = workbook.create_sheet(f"Noise {noise_level:g}")
        header = ["Split", "Category", "Task Name"]
        for ode, _ in comparable:
            header.extend([f"ODE {ode.step}", f"CPS {ode.step}", f"Delta {ode.step}"])
        sheet.append(header)
        metadata = comparable[0][1].task_meta
        for task in sorted(metadata, key=lambda name: _task_sort_key(name, metadata)):
            split, category = metadata[task]
            row: list[Any] = [split.replace("_", "-"), category, task]
            for ode, cps in comparable:
                ode_score = ode.task_scores[task]
                cps_score = cps.task_scores[task]
                row.extend([ode_score, cps_score, cps_score - ode_score])
            sheet.append(row)
        score_columns = list(range(4, len(header) + 1))
        _format_sheet(sheet, [18, 18, 78, *([14] * (len(header) - 3))], score_columns)

    coverage = workbook.create_sheet("Coverage")
    coverage.append(["Run", "Checkpoint Step", "CPS Noise Level", "Status"])
    for run in cps_runs:
        coverage.append([run.name, run.step, run.noise_level, "complete"])
    for name, reason in skipped:
        match = DANCE_RUN_RE.fullmatch(name)
        if match and match.group("noise") is not None:
            coverage.append([name, int(match.group("step")), float(match.group("noise")), reason])
    _format_sheet(coverage, [72, 18, 18, 48])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def write_ode_step_trend(runs: list[RunResult], output_path: Path) -> None:
    ode_runs = sorted(
        (run for run in runs if run.family == "DanceGRPO" and run.sampler == "ODE" and run.step is not None),
        key=lambda run: run.step or -1,
    )
    if not ode_runs:
        raise ValueError("No DanceGRPO ODE runs found")
    baseline = ode_runs[0]

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "ODE by Step"
    sheet.append(
        [
            "Checkpoint Step",
            "Overall",
            "In-Domain",
            "Out-of-Domain",
            "Overall Delta vs Previous",
            f"Overall Delta vs {baseline.step}",
            f"In-Domain Delta vs {baseline.step}",
            f"Out-of-Domain Delta vs {baseline.step}",
            *CATEGORY_ORDER,
        ]
    )
    previous: RunResult | None = None
    for run in ode_runs:
        sheet.append(
            [
                run.step,
                run.overall,
                run.in_domain,
                run.out_of_domain,
                None if previous is None else run.overall - previous.overall,
                run.overall - baseline.overall,
                run.in_domain - baseline.in_domain,
                run.out_of_domain - baseline.out_of_domain,
                *(_category_score(run, category) for category in CATEGORY_ORDER),
            ]
        )
        previous = run
    _format_sheet(sheet, [18, 14, 14, 16, 26, 24, 26, 28, 16, 16, 18, 16, 16], list(range(2, 14)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def write_cps_step_trend(runs: list[RunResult], output_path: Path, noise_level: float) -> None:
    baselines = sorted(
        (run for run in runs if run.family == "SFT" and run.sampler == "ODE"),
        key=lambda run: run.name,
    )
    if not baselines:
        raise ValueError("No SFT ODE baseline found")
    baseline = baselines[0]
    cps_runs = sorted(
        (
            run
            for run in runs
            if run.family == "DanceGRPO"
            and run.sampler == "CPS"
            and run.noise_level == noise_level
            and run.step is not None
        ),
        key=lambda run: run.step or -1,
    )
    if not cps_runs:
        raise ValueError(f"No DanceGRPO CPS noise {noise_level:g} runs found")
    baseline_tasks = set(baseline.task_scores)
    for run in cps_runs:
        if set(run.task_scores) != baseline_tasks:
            raise ValueError(f"{run.name}: task set does not match the SFT ODE baseline")

    metadata = baseline.task_meta
    tasks = sorted(baseline_tasks, key=lambda name: _task_sort_key(name, metadata))
    step_runs = [(run.step, run) for run in cps_runs]

    workbook = Workbook()
    score_sheet = workbook.active
    score_sheet.title = "Task Scores"
    score_sheet.append(
        [
            "Split",
            "Category",
            "Task Name",
            "ODE Baseline (Step 0)",
            *(f"CPS {step}" for step, _ in step_runs),
        ]
    )
    for task in tasks:
        split, category = metadata[task]
        score_sheet.append(
            [
                split.replace("_", "-"),
                category,
                task,
                baseline.task_scores[task],
                *(run.task_scores[task] for _, run in step_runs),
            ]
        )
    _format_sheet(score_sheet, [18, 18, 78, *([18] * (len(step_runs) + 1))], list(range(4, len(step_runs) + 5)))

    baseline_delta_sheet = workbook.create_sheet("Delta vs Baseline")
    baseline_delta_sheet.append(
        ["Split", "Category", "Task Name", *(f"CPS {step} - ODE Baseline" for step, _ in step_runs)]
    )
    for task in tasks:
        split, category = metadata[task]
        baseline_score = baseline.task_scores[task]
        baseline_delta_sheet.append(
            [
                split.replace("_", "-"),
                category,
                task,
                *(run.task_scores[task] - baseline_score for _, run in step_runs),
            ]
        )
    _format_sheet(
        baseline_delta_sheet,
        [18, 18, 78, *([24] * len(step_runs))],
        list(range(4, len(step_runs) + 4)),
    )

    previous_delta_sheet = workbook.create_sheet("Delta vs Previous")
    previous_headers = []
    previous_run = baseline
    previous_step: int | str = "ODE Baseline"
    for step, run in step_runs:
        previous_headers.append(f"CPS {step} - {previous_step}")
        previous_step = f"CPS {step}"
        previous_run = run
    previous_delta_sheet.append(["Split", "Category", "Task Name", *previous_headers])
    for task in tasks:
        split, category = metadata[task]
        deltas = []
        previous_run = baseline
        for _, run in step_runs:
            deltas.append(run.task_scores[task] - previous_run.task_scores[task])
            previous_run = run
        previous_delta_sheet.append([split.replace("_", "-"), category, task, *deltas])
    _format_sheet(
        previous_delta_sheet,
        [18, 18, 78, *([24] * len(step_runs))],
        list(range(4, len(step_runs) + 4)),
    )

    summary_sheet = workbook.create_sheet("Task Summary")
    summary_sheet.append(
        [
            "Split",
            "Category",
            "Task Name",
            "ODE Baseline",
            "Best Score",
            "Best Step",
            "Best Delta vs Baseline",
            "Final Score",
            "Final Delta vs Baseline",
            "Improved Checkpoints",
        ]
    )
    final_step, final_run = step_runs[-1]
    for task in tasks:
        split, category = metadata[task]
        baseline_score = baseline.task_scores[task]
        candidates = [(0, baseline_score), *((step, run.task_scores[task]) for step, run in step_runs)]
        best_step, best_score = max(candidates, key=lambda item: (item[1], -int(item[0])))
        final_score = final_run.task_scores[task]
        summary_sheet.append(
            [
                split.replace("_", "-"),
                category,
                task,
                baseline_score,
                best_score,
                best_step,
                best_score - baseline_score,
                final_score,
                final_score - baseline_score,
                sum(run.task_scores[task] > baseline_score for _, run in step_runs),
            ]
        )
    _format_sheet(summary_sheet, [18, 18, 78, 18, 16, 14, 24, 16, 24, 22], [4, 5, 7, 8, 9])

    aggregate_sheet = workbook.create_sheet("Aggregate")
    aggregate_sheet.append(
        [
            "Training Step",
            "Run",
            "Sampler",
            "CPS Noise Level",
            "Overall",
            "In-Domain",
            "Out-of-Domain",
            "Overall Delta vs Previous",
            "Overall Delta vs Baseline",
            "In-Domain Delta vs Baseline",
            "Out-of-Domain Delta vs Baseline",
            *CATEGORY_ORDER,
        ]
    )
    previous = baseline
    ordered_runs = [(0, baseline), *((run.step, run) for run in cps_runs)]
    for step, run in ordered_runs:
        is_baseline = run is baseline
        aggregate_sheet.append(
            [
                step,
                run.name,
                "ODE baseline" if is_baseline else "CPS",
                None if is_baseline else run.noise_level,
                run.overall,
                run.in_domain,
                run.out_of_domain,
                None if is_baseline else run.overall - previous.overall,
                run.overall - baseline.overall,
                run.in_domain - baseline.in_domain,
                run.out_of_domain - baseline.out_of_domain,
                *(_category_score(run, category) for category in CATEGORY_ORDER),
            ]
        )
        previous = run
    _format_sheet(
        aggregate_sheet,
        [18, 72, 16, 18, 14, 14, 16, 26, 26, 28, 30, 16, 16, 18, 16, 16],
        list(range(5, 17)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def generate_reports(
    root: Path,
    output_dir: Path,
    expected_samples: int = 500,
    expected_evalkit_source_sha256: str | None = None,
) -> tuple[Path, Path, Path, Path, Path, int, int]:
    runs, skipped = discover_runs(
        root,
        expected_samples=expected_samples,
        expected_evalkit_source_sha256=expected_evalkit_source_sha256,
    )
    if not runs:
        raise ValueError(f"No complete standard results found under {root}")
    all_runs_path = output_dir / "all_run_summary.xlsx"
    cps_path = output_dir / "cps_task_changes_vs_ode.xlsx"
    ode_path = output_dir / "ode_scores_by_training_step.xlsx"
    cps_trend_path = output_dir / "cps_0p3_scores_by_training_step.xlsx"
    cps_0p7_trend_path = output_dir / "cps_0p7_scores_by_training_step.xlsx"
    write_all_run_summary(runs, skipped, all_runs_path)
    write_cps_task_changes(runs, skipped, cps_path)
    write_ode_step_trend(runs, ode_path)
    write_cps_step_trend(runs, cps_trend_path, noise_level=0.3)
    write_cps_step_trend(runs, cps_0p7_trend_path, noise_level=0.7)
    return all_runs_path, cps_path, ode_path, cps_trend_path, cps_0p7_trend_path, len(runs), len(skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed VBVR-Pro main_v2 evaluations into Excel files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("storage/eval_out/vbvr_pro_main_v2_evalkit_4cc7d028"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument(
        "--expected-evalkit-source-sha256",
        default="4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8",
        help="Only include complete runs with this exact scorer-contract fingerprint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root / "reports"
    paths = generate_reports(
        args.root,
        output_dir,
        expected_samples=args.expected_samples,
        expected_evalkit_source_sha256=args.expected_evalkit_source_sha256,
    )
    for path in paths[:5]:
        print(path)
    print(f"Complete runs: {paths[5]}; skipped/incomplete runs: {paths[6]}")


if __name__ == "__main__":
    main()
