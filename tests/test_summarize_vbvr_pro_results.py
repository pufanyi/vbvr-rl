import json
from pathlib import Path

from openpyxl import load_workbook
from pytest import approx, raises

from src.cli.summarize_vbvr_pro_results import RESULT_RELATIVE_PATH, _load_run, generate_reports
from src.eval.evaluation_provenance import build_manifest, write_manifest


def _write_result(root: Path, run_name: str, task_a: tuple[float, float], task_b: tuple[float, float]) -> None:
    samples = []
    for task_name, split, category, scores in (
        ("G-1_task", "In_Domain", "Abstraction", task_a),
        ("G-2_task", "Out_of_Domain", "Knowledge", task_b),
    ):
        for index, score in enumerate(scores):
            samples.append(
                {
                    "task_name": task_name,
                    "split": split,
                    "category": category,
                    "score": score,
                    "video_file": f"{index}.mp4",
                    "error": None,
                }
            )
    task_scores = {"G-1_task": sum(task_a) / 2, "G-2_task": sum(task_b) / 2}
    in_score = task_scores["G-1_task"]
    out_score = task_scores["G-2_task"]
    result = {
        "samples": samples,
        "summary": {
            "In_Domain": {"mean_score": in_score, "num_samples": 2},
            "Out_of_Domain": {"mean_score": out_score, "num_samples": 2},
            "overall": {
                "mean_score": (in_score + out_score) / 2,
                "num_samples": 4,
                "by_task": task_scores,
                "by_category": {"Abstraction": in_score, "Knowledge": out_score},
            },
        },
    }
    path = root / run_name / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    provenance = build_manifest(
        stage="vbvr-pro-score",
        values={
            "state": "complete",
            "evalkit_revision": "a" * 40,
            "evalkit_source_sha256": "b" * 64,
        },
        files={},
        trees={},
        output_files={"result": str(path)},
    )
    write_manifest(root / run_name / "score-provenance.json", provenance)


def test_load_run_rejects_result_replaced_after_provenance(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_name = "dancegrpo_vbvr_pro_5b_checkpoint-300"
    _write_result(root, run_name, (0.2, 0.4), (0.6, 0.8))
    result_path = root / run_name / RESULT_RELATIVE_PATH
    result_path.write_text(result_path.read_text() + "\n")

    with raises(ValueError, match="recorded artifact changed"):
        _load_run(root / run_name, expected_samples=4)


def test_generate_reports(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_result(root, "sft_vbvr_5b_checkpoint-epoch1", (0.4, 0.6), (0.5, 0.7))
    _write_result(root, "dancegrpo_vbvr_pro_5b_checkpoint-300", (0.2, 0.4), (0.6, 0.8))
    _write_result(root, "dancegrpo_vbvr_pro_5b_checkpoint-300-cps-noise-0.3", (0.5, 0.7), (0.4, 0.6))
    _write_result(root, "dancegrpo_vbvr_pro_5b_checkpoint-300-cps-noise-0.7", (0.6, 0.8), (0.3, 0.5))
    _write_result(root, "dancegrpo_vbvr_pro_5b_checkpoint-600", (0.3, 0.5), (0.7, 0.9))
    (root / "dancegrpo_vbvr_pro_5b_checkpoint-600-cps-noise-0.3").mkdir(parents=True)
    output = tmp_path / "reports"

    all_runs, cps, ode, cps_trend, cps_0p7_trend, complete_count, skipped_count = generate_reports(
        root, output, expected_samples=4
    )

    assert complete_count == 5
    assert skipped_count == 1
    all_book = load_workbook(all_runs, data_only=True)
    assert all_book["All Runs"].max_row == 6
    assert all_book["Skipped"].max_row == 2

    cps_book = load_workbook(cps, data_only=True)
    assert cps_book.sheetnames == ["All Deltas", "Noise 0.3", "Noise 0.7", "Coverage"]
    first_delta = list(cps_book["All Deltas"].iter_rows(min_row=2, max_row=2, values_only=True))[0]
    assert first_delta[6:] == approx((0.3, 0.6, 0.3))
    coverage = list(cps_book["Coverage"].iter_rows(min_row=2, values_only=True))
    assert any(row[0].endswith("checkpoint-600-cps-noise-0.3") and row[3] != "complete" for row in coverage)

    ode_book = load_workbook(ode, data_only=True)
    rows = list(ode_book["ODE by Step"].iter_rows(min_row=2, values_only=True))
    assert [row[0] for row in rows] == [300, 600]
    assert rows[0][4] is None
    assert rows[1][4] == approx(0.1)

    cps_trend_book = load_workbook(cps_trend, data_only=True)
    assert cps_trend_book.sheetnames == [
        "Task Scores",
        "Delta vs Baseline",
        "Delta vs Previous",
        "Task Summary",
        "Aggregate",
    ]
    task_score = next(cps_trend_book["Task Scores"].iter_rows(min_row=2, max_row=2, values_only=True))
    assert task_score[:3] == ("In-Domain", "Abstraction", "G-1_task")
    assert task_score[3:] == approx((0.5, 0.6))
    baseline_delta = next(cps_trend_book["Delta vs Baseline"].iter_rows(min_row=2, max_row=2, values_only=True))
    assert baseline_delta[3:] == approx((0.1,))
    task_summary = next(cps_trend_book["Task Summary"].iter_rows(min_row=2, max_row=2, values_only=True))
    assert task_summary[3:9] == approx((0.5, 0.6, 300, 0.1, 0.6, 0.1))
    trend_rows = list(cps_trend_book["Aggregate"].iter_rows(min_row=2, values_only=True))
    assert [row[0] for row in trend_rows] == [0, 300]
    assert trend_rows[0][2:4] == ("ODE baseline", None)
    assert trend_rows[1][2:4] == ("CPS", 0.3)
    assert trend_rows[1][8] == approx(0.0)

    cps_0p7_book = load_workbook(cps_0p7_trend, data_only=True)
    trend_0p7_rows = list(cps_0p7_book["Aggregate"].iter_rows(min_row=2, values_only=True))
    assert [row[0] for row in trend_0p7_rows] == [0, 300]
    assert trend_0p7_rows[1][2:4] == ("CPS", 0.7)
