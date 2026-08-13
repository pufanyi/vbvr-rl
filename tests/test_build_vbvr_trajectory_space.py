import hashlib
import json
from pathlib import Path

import pytest

from scripts.eval.vbvr_pro.build_vbvr_trajectory_space import MODELS, SAMPLERS, build_space
from scripts.eval.vbvr_pro.upload_vbvr_trajectory_steps import build_upload_items, chunks, model_mapping


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_fixture(root: Path) -> list[Path]:
    samples = [
        Path("In-Domain_50/G-1_first-task_data-generator/00000"),
        Path("Out-of-Domain_50/O-2_second-task_data-generator/00001"),
    ]
    for model_index, model in enumerate(MODELS):
        for sampler_index, sampler in enumerate(SAMPLERS):
            cell = root / f"{model.source_prefix}-{sampler.source_suffix}"
            formal_run_root = root.parent / "formal-results" / cell.name
            formal_final_root = formal_run_root / "generated_512x512x81"
            _write_json(
                cell / "cell_manifest.json",
                {
                    "state": "complete",
                    "sample_count": len(samples),
                    "completed_count": len(samples),
                    "formal_final_root": str(formal_final_root.resolve()),
                },
            )
            cell_index = model_index * len(SAMPLERS) + sampler_index
            score_values = [cell_index / 20, (cell_index + 1) / 20]
            score_samples = []
            for sample_index, relative in enumerate(samples):
                sample_root = cell / relative
                formal_video = formal_final_root / relative.parent / f"{relative.name}.mp4"
                formal_video.parent.mkdir(parents=True, exist_ok=True)
                formal_video.write_bytes(f"formal:{model.id}:{sampler.id}:{sample_index}".encode())
                previews = [
                    {
                        "display_step": step + 1,
                        "file_index": step,
                        "kind": "final_latent" if step == 29 else "predicted_clean_x0",
                        "source_sigma": 1.0 - step / 30,
                        "output_sigma": 0.0,
                    }
                    for step in range(30)
                ]
                _write_json(
                    sample_root / "manifest.json",
                    {
                        "summary": {
                            "name": relative.as_posix(),
                            "prompt": f"Prompt {sample_index}",
                        },
                        "seed": sample_index,
                        "step_previews": previews,
                        "formal_final_binding": {
                            "source": str(formal_video.resolve()),
                            "sha256": "a" * 64,
                            "bound_outputs": [
                                str((sample_root / "final_00.mp4").resolve()),
                                str((sample_root / "step_29.mp4").resolve()),
                            ],
                        },
                    },
                )
                (sample_root / "steps_grid.mp4").write_bytes(f"grid:{model.id}:{sampler.id}".encode())
                (sample_root / "final_00.mp4").write_bytes(f"final:{model.id}:{sampler.id}".encode())
                for step in range(30):
                    (sample_root / f"step_{step:02d}.mp4").write_bytes(f"step:{step}:{model.id}:{sampler.id}".encode())
                domain_folder, task_name, sample_name = relative.parts
                score = score_values[sample_index]
                score_samples.append(
                    {
                        "video_path": str(formal_video.resolve()),
                        "video_file": f"{sample_name}.mp4",
                        "task_name": task_name,
                        "split": "In_Domain" if domain_folder == "In-Domain_50" else "Out_of_Domain",
                        "category": "fixture",
                        "folder": domain_folder,
                        "score": score,
                        "dimensions": {"task_specific": score},
                        "error": None,
                    }
                )

            def score_group(values: list[float]) -> dict[str, object]:
                return {
                    "scores": values,
                    "mean_score": sum(values) / len(values),
                    "num_samples": len(values),
                }

            result_path = formal_run_root / "scores/eval_fixture_vbvr_results.json"
            _write_json(
                result_path,
                {
                    "samples": list(reversed(score_samples)),
                    "summary": {
                        "In_Domain": score_group(score_values[:1]),
                        "Out_of_Domain": score_group(score_values[1:]),
                        "overall": score_group(score_values),
                    },
                },
            )
            result_bytes = result_path.read_bytes()
            _write_json(
                formal_run_root / "score-provenance.json",
                {
                    "stage": "vbvr-pro-score",
                    "values": {
                        "state": "complete",
                        "evalkit_revision": "e140038f2aee76ca518f464755fa8bc19b783ba5",
                        "evalkit_revision_actual": (
                            "e140038f2aee76ca518f464755fa8bc19b783ba5" if model.id == "baseline" else "unavailable"
                        ),
                        "evalkit_source_sha256": "4" * 64,
                        "scorer_dependencies": json.dumps({"contract": "fixture-runtime-v1", "sha256": "9" * 64}),
                    },
                    "output_files": {
                        "result": {
                            "path": str(result_path.resolve()),
                            "sha256": hashlib.sha256(result_bytes).hexdigest(),
                            "size": len(result_bytes),
                        }
                    },
                },
            )
    return samples


def test_build_space_indexes_and_materializes_all_matched_cells(tmp_path):
    source = tmp_path / "source"
    samples = _build_fixture(source)
    output = tmp_path / "space"
    dataset = tmp_path / "dataset"
    step_datasets = {
        "baseline": tmp_path / "baseline-steps",
        "checkpoint-2200": tmp_path / "checkpoint-2200-steps",
    }
    prefix = "https://huggingface.co/datasets/example/trajectories/resolve/main/videos"
    step_prefixes = {
        "baseline": "https://huggingface.co/datasets/example/baseline-steps/resolve/main/videos",
        "checkpoint-2200": "https://huggingface.co/datasets/example/2200-steps/resolve/main/videos",
    }

    index = build_space(
        trajectory_root=source,
        output_root=output,
        dataset_output_root=dataset,
        media_url_prefix=prefix,
        step_dataset_output_roots=step_datasets,
        step_media_url_prefixes=step_prefixes,
        skip_videos=False,
        copy_videos=False,
        expected_samples=2,
        expected_tasks=2,
    )

    assert index["sampleCount"] == 2
    assert index["schemaVersion"] == 3
    assert index["taskCount"] == 2
    assert index["samplesPerTask"] == 1
    assert index["cellCount"] == 12
    assert index["videoCount"] == 768
    assert index["overviewVideoCount"] == 48
    assert index["originalStepVideoCount"] == 720
    assert index["mediaUrlPrefix"] == prefix
    assert index["stepMediaUrlPrefixes"] == step_prefixes
    assert {sample["id"] for sample in index["samples"]} == {sample.as_posix() for sample in samples}
    assert len(index["samplers"]) == 6
    assert all(len(sampler["schedule"]) == 30 for sampler in index["samplers"])
    assert index["scoreContract"]["scoreCount"] == 24
    assert index["scoreContract"]["appliesTo"] == "final-only"
    assert index["scores"]["baseline--cps-0.1"] == [0.0, 0.05]
    assert index["scores"]["checkpoint-2200--unipc"] == [0.55, 0.6]
    assert index["cells"][0]["scoreSummary"]["overall"] == pytest.approx(0.025)
    assert all(cell["scoreSummary"]["errorCount"] == 0 for cell in index["cells"])

    generated = json.loads((output / "data/index.json").read_text(encoding="utf-8"))
    assert generated == index
    assert (output / "index.html").is_file()
    assert (dataset / "README.md").is_file()
    assert len(list((dataset / "videos").rglob("*.mp4"))) == 48
    assert len(list((step_datasets["baseline"] / "videos").rglob("*.mp4"))) == 360
    assert len(list((step_datasets["checkpoint-2200"] / "videos").rglob("*.mp4"))) == 360
    source_grid = source / "baseline-cps0p1" / samples[0] / "steps_grid.mp4"
    dataset_grid = dataset / "videos/baseline--cps-0.1" / samples[0] / "steps_grid.mp4"
    assert source_grid.samefile(dataset_grid)
    source_step = source / "2200-unipc" / samples[-1] / "step_17.mp4"
    dataset_step = step_datasets["checkpoint-2200"] / "videos/checkpoint-2200--unipc" / samples[-1] / "step_17.mp4"
    assert source_step.samefile(dataset_step)


def test_build_space_rejects_a_cell_with_a_missing_sample(tmp_path):
    source = tmp_path / "source"
    samples = _build_fixture(source)
    missing_root = source / "2200-unipc" / samples[-1]
    (missing_root / "steps_grid.mp4").unlink()

    with pytest.raises(ValueError, match="expected 2 trajectory grids, found 1"):
        build_space(
            trajectory_root=source,
            output_root=tmp_path / "space",
            dataset_output_root=None,
            media_url_prefix="https://example.test/videos",
            skip_videos=True,
            copy_videos=False,
            expected_samples=2,
            expected_tasks=2,
        )


def test_build_space_rejects_a_missing_native_step_video(tmp_path):
    source = tmp_path / "source"
    samples = _build_fixture(source)
    missing = source / "baseline-euler" / samples[0] / "step_12.mp4"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="step_12.mp4"):
        build_space(
            trajectory_root=source,
            output_root=tmp_path / "space",
            dataset_output_root=None,
            media_url_prefix="https://example.test/videos",
            skip_videos=True,
            copy_videos=False,
            expected_samples=2,
            expected_tasks=2,
        )


def test_build_space_rejects_a_score_result_changed_after_provenance(tmp_path):
    source = tmp_path / "source"
    _build_fixture(source)
    result_path = tmp_path / "formal-results/baseline-cps0p1/scores/eval_fixture_vbvr_results.json"
    result_path.write_text(f"{result_path.read_text(encoding='utf-8')}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="result SHA-256 does not match"):
        build_space(
            trajectory_root=source,
            output_root=tmp_path / "space",
            dataset_output_root=None,
            media_url_prefix="https://example.test/videos",
            skip_videos=True,
            copy_videos=False,
            expected_samples=2,
            expected_tasks=2,
        )


def test_native_step_upload_plan_is_split_by_model(tmp_path):
    source = tmp_path / "source"
    samples = _build_fixture(source)

    items = build_upload_items(source, model_id="baseline", expected_samples=2)

    assert len(items) == 360
    assert len(chunks(items, 100)) == 4
    assert all(item.remote_path.startswith("videos/baseline--") for item in items)
    assert items[0].remote_path.endswith(f"{samples[0].as_posix()}/step_00.mp4")
    assert model_mapping(
        ["baseline=example/custom-baseline"],
        defaults={"baseline": "default/a", "checkpoint-2200": "default/b"},
    ) == {
        "baseline": "example/custom-baseline",
        "checkpoint-2200": "default/b",
    }
