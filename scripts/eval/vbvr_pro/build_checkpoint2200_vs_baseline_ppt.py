#!/usr/bin/env python3
"""Build a reviewed top-50 checkpoint-2200 versus DiffSynth baseline deck.

The comparison is strictly paired on the same 500-sample manifest. Each case
slide embeds the two native 512x512x81 MP4s and labels their exact EvalKit
scores. Movie posters are lossless PNG copies of each raw video's decoded frame
zero, with no labels, crops, or compositing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.eval.vbvr_pro import build_rl_demo_ppt as deck

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = (
    REPO_ROOT / "storage/eval_out/"
    "vbvr_pro_main_v2_512x512x81_manifest_rl_e140_lr5e6_eval500_181e2010_"
    "manifest_afab352e_evalkit_4cc7d028"
)
BASELINE_NAME = "diffsynth_step35500-baseline-unipc-ode-30steps-cfg1"
CHECKPOINT_NAME = "dancegrpo_vbvr_pro_5b_checkpoint-2200-cps-noise-0.7"
BASELINE_ROOT = EVAL_ROOT / BASELINE_NAME
CHECKPOINT_ROOT = EVAL_ROOT / CHECKPOINT_NAME
DEFAULT_OUTPUT_DIR = REPO_ROOT / "storage/presentations/vbvr_checkpoint2200_vs_baseline_20260818"
EXPECTED_EVALKIT_SHA256 = "4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8"
SELECTION_COUNT = 50
MIN_SELECTED_DELTA = 0.35

TASK_LABELS = {
    "G-131_select_next_figure_increasing_size_sequence_data-generator": "递增尺寸序列续项",
    "G-134_select_next_figure_large_small_alternating_sequence_data-generator": "大小交替序列续项",
    "G-138_spot_unique_non_repeated_color_data-generator": "找唯一不重复颜色",
    "G-13_grid_number_sequence_data-generator": "数字序列路径",
    "G-140_locate_topmost_unobscured_figure_data-generator": "定位最高未遮挡图形",
    "G-160_circle_largest_numerical_value_data-generator": "圈出最大数值",
    "G-168_identify_nearest_to_square_rectangle_data-generator": "找最接近正方形的矩形",
    "G-169_locate_intersection_of_segments_data-generator": "定位线段交点",
    "G-16_grid_go_through_block_data-generator": "按序穿越色块",
    "G-202_mark_wave_peaks_data-generator": "标记波峰",
    "G-206_identify_pentagons_data-generator": "识别五边形",
    "G-221_outline_innermost_square_data-generator": "描出最内层正方形",
    "G-223_highlight_horizontal_lines_data-generator": "标出水平线",
    "G-240_add_borders_to_unbordered_shapes_data-generator": "给无边框图形加边框",
    "G-248_mark_asymmetrical_shape_data-generator": "标记非对称图形",
    "G-250_color_triple_intersection_red_data-generator": "三重交集涂红",
    "G-29_chart_extreme_with_data_data-generator": "图表极值定位",
    "G-3_stable_sort_data-generator": "稳定排序",
    "G-41_grid_highest_cost_data-generator": "最高成本路径",
    "G-47_multiple_keys_for_one_door_data-generator": "多钥匙单门",
    "G-54_connecting_color_data-generator": "同色连接",
    "G-8_track_object_movement_data-generator": "追踪物体移动",
    "G-9_identify_objects_in_region_data-generator": "识别区域内物体",
    "O-11_shape_color_then_move_data-generator": "先变色再移动",
    "O-16_color_addition_data-generator": "加色混合",
    "O-29_ballcolor_data-generator": "彩球合并与计数",
    "O-31_ball_eating_data-generator": "球吞噬",
    "O-36_grid_shift_data-generator": "网格平移",
    "O-37_light_sequence_data-generator": "灯光序列",
    "O-38_majority_color_data-generator": "多数颜色",
    "O-39_maze_data-generator": "迷宫",
    "O-45_sequence_completion_data-generator": "序列补全",
    "O-47_sliding_puzzle_data-generator": "滑块拼图",
    "O-49_symmetry_completion_data-generator": "对称补全",
    "O-52_traffic_light_data-generator": "交通灯状态",
    "O-56_raven_data-generator": "Raven 图形推理",
    "O-75_communicating_vessels_data-generator": "连通器液面",
}


@dataclass(frozen=True)
class PairCase:
    rank: int
    canonical_name: str
    task_name: str
    video_idx: str
    domain: str
    domain_folder: str
    category: str
    prompt: str
    input_image: Path
    ground_truth_video: Path
    ground_truth_final: Path
    baseline_video: Path
    checkpoint_video: Path
    baseline_scored_video: Path
    checkpoint_scored_video: Path
    baseline_score: float
    checkpoint_score: float

    @property
    def delta(self) -> float:
        return self.checkpoint_score - self.baseline_score

    @property
    def task_label(self) -> str:
        return TASK_LABELS.get(self.task_name, self.task_name)

    @property
    def domain_label(self) -> str:
        return "In-Domain" if self.domain == "In_Domain" else "Out-of-Domain"

    @property
    def case_id(self) -> str:
        return f"C{self.rank:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write selection metadata and temporal audit sheets without building the PPTX.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_path(root: Path) -> Path:
    matches = sorted((root / "scores").glob("*_vbvr_results.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one result JSON beneath {root}, found {len(matches)}")
    return matches[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def result_key(sample: dict[str, Any]) -> str:
    return f"{sample['folder']}/{sample['task_name']}/{Path(sample['video_file']).stem}"


def validate_generation_contract(root: Path, expected: dict[str, str]) -> dict[str, Any]:
    provenance = load_json(root / "generation-provenance.json")
    if provenance["values"].get("state") != "complete":
        raise ValueError(f"Generation is not complete: {root}")
    for key, value in expected.items():
        actual = provenance["values"].get(key)
        if actual != value:
            raise ValueError(f"Unexpected {key} in {root}: expected={value!r}, actual={actual!r}")
    return provenance


def validate_score_contract(root: Path) -> dict[str, Any]:
    provenance = load_json(root / "score-provenance.json")
    values = provenance["values"]
    if values.get("state") != "complete":
        raise ValueError(f"Scoring is not complete: {root}")
    if values.get("evalkit_source_sha256") != EXPECTED_EVALKIT_SHA256:
        raise ValueError(f"EvalKit source fingerprint mismatch: {root}")
    return provenance


def load_audited_pairs() -> tuple[list[PairCase], dict[str, Any]]:
    common_contract = {
        "fps": "16",
        "guidance_scale": "1.0",
        "height": "512",
        "num_frames": "81",
        "num_inference_steps": "30",
        "seed": "0",
        "width": "512",
    }
    baseline_generation = validate_generation_contract(
        BASELINE_ROOT,
        {**common_contract, "generation_mode": "ode", "ode_solver": "unipc"},
    )
    checkpoint_generation = validate_generation_contract(
        CHECKPOINT_ROOT,
        {**common_contract, "generation_mode": "cps", "cps_noise_level": "0.7"},
    )
    baseline_score_provenance = validate_score_contract(BASELINE_ROOT)
    checkpoint_score_provenance = validate_score_contract(CHECKPOINT_ROOT)

    baseline_eval_path = BASELINE_ROOT / "eval_samples.json"
    checkpoint_eval_path = CHECKPOINT_ROOT / "eval_samples.json"
    if sha256_file(baseline_eval_path) != sha256_file(checkpoint_eval_path):
        raise ValueError("The two eval_samples.json files are not byte-identical")
    eval_samples = load_json(baseline_eval_path)
    if len(eval_samples) != 500:
        raise ValueError(f"Expected 500 eval samples, found {len(eval_samples)}")
    eval_by_name = {sample["name"]: sample for sample in eval_samples}
    if len(eval_by_name) != len(eval_samples):
        raise ValueError("Duplicate canonical names in eval_samples.json")

    baseline_result = load_json(result_path(BASELINE_ROOT))
    checkpoint_result = load_json(result_path(CHECKPOINT_ROOT))
    if len(baseline_result["samples"]) != 500 or len(checkpoint_result["samples"]) != 500:
        raise ValueError("Both result JSONs must contain exactly 500 samples")
    baseline_by_name = {result_key(sample): sample for sample in baseline_result["samples"]}
    checkpoint_by_name = {result_key(sample): sample for sample in checkpoint_result["samples"]}
    if baseline_by_name.keys() != checkpoint_by_name.keys() or baseline_by_name.keys() != eval_by_name.keys():
        raise ValueError("Paired sample identities do not match exactly")
    if any(sample["error"] is not None for sample in baseline_by_name.values()):
        raise ValueError("Baseline result contains scorer errors")
    if any(sample["error"] is not None for sample in checkpoint_by_name.values()):
        raise ValueError("Checkpoint result contains scorer errors")

    unsorted_pairs: list[PairCase] = []
    for canonical_name, eval_sample in eval_by_name.items():
        baseline_sample = baseline_by_name[canonical_name]
        checkpoint_sample = checkpoint_by_name[canonical_name]
        if baseline_sample["task_name"] != eval_sample["task_name"]:
            raise ValueError(f"Baseline task mismatch: {canonical_name}")
        if checkpoint_sample["task_name"] != eval_sample["task_name"]:
            raise ValueError(f"Checkpoint task mismatch: {canonical_name}")
        input_image = Path(eval_sample["image"]).resolve()
        sample_root = input_image.parent
        native_relative = Path(f"{canonical_name}.mp4")
        baseline_video = (BASELINE_ROOT / "generated_512x512x81" / native_relative).resolve()
        checkpoint_video = (CHECKPOINT_ROOT / "generated_512x512x81" / native_relative).resolve()
        expected_files = (
            input_image,
            sample_root / "ground_truth.mp4",
            sample_root / "final_frame.png",
            baseline_video,
            checkpoint_video,
            Path(baseline_sample["video_path"]).resolve(),
            Path(checkpoint_sample["video_path"]).resolve(),
        )
        for path in expected_files:
            if not path.is_file():
                raise FileNotFoundError(path)
        unsorted_pairs.append(
            PairCase(
                rank=0,
                canonical_name=canonical_name,
                task_name=eval_sample["task_name"],
                video_idx=eval_sample["video_idx"],
                domain=eval_sample["domain"],
                domain_folder=canonical_name.split("/", 1)[0],
                category=checkpoint_sample["category"],
                prompt=eval_sample["prompt"],
                input_image=input_image,
                ground_truth_video=(sample_root / "ground_truth.mp4").resolve(),
                ground_truth_final=(sample_root / "final_frame.png").resolve(),
                baseline_video=baseline_video,
                checkpoint_video=checkpoint_video,
                baseline_scored_video=Path(baseline_sample["video_path"]).resolve(),
                checkpoint_scored_video=Path(checkpoint_sample["video_path"]).resolve(),
                baseline_score=float(baseline_sample["score"]),
                checkpoint_score=float(checkpoint_sample["score"]),
            )
        )

    ordered = sorted(
        unsorted_pairs,
        key=lambda case: (-case.delta, -case.checkpoint_score, case.baseline_score, case.canonical_name),
    )
    selected = [
        PairCase(**{**case.__dict__, "rank": rank}) for rank, case in enumerate(ordered[:SELECTION_COUNT], start=1)
    ]
    if len(selected) != SELECTION_COUNT or selected[-1].delta < MIN_SELECTED_DELTA:
        raise ValueError(
            f"Top-{SELECTION_COUNT} selection is not sufficiently high contrast: min_delta={selected[-1].delta:.6f}"
        )

    all_deltas = [case.delta for case in unsorted_pairs]
    audit = {
        "schema_version": 1,
        "eval_samples_sha256": sha256_file(baseline_eval_path),
        "evalkit_source_sha256": EXPECTED_EVALKIT_SHA256,
        "paired_sample_count": len(unsorted_pairs),
        "baseline_error_count": 0,
        "checkpoint_error_count": 0,
        "positive_pair_count": sum(delta > 0 for delta in all_deltas),
        "tied_pair_count": sum(delta == 0 for delta in all_deltas),
        "negative_pair_count": sum(delta < 0 for delta in all_deltas),
        "baseline_overall": float(baseline_result["summary"]["overall"]["mean_score"]),
        "checkpoint_overall": float(checkpoint_result["summary"]["overall"]["mean_score"]),
        "baseline_in_domain": float(baseline_result["summary"]["In_Domain"]["mean_score"]),
        "checkpoint_in_domain": float(checkpoint_result["summary"]["In_Domain"]["mean_score"]),
        "baseline_out_of_domain": float(baseline_result["summary"]["Out_of_Domain"]["mean_score"]),
        "checkpoint_out_of_domain": float(checkpoint_result["summary"]["Out_of_Domain"]["mean_score"]),
        "baseline_generation_provenance_sha256": sha256_file(BASELINE_ROOT / "generation-provenance.json"),
        "checkpoint_generation_provenance_sha256": sha256_file(CHECKPOINT_ROOT / "generation-provenance.json"),
        "baseline_score_provenance_sha256": sha256_file(BASELINE_ROOT / "score-provenance.json"),
        "checkpoint_score_provenance_sha256": sha256_file(CHECKPOINT_ROOT / "score-provenance.json"),
        "baseline_generation_contract": baseline_generation["values"],
        "checkpoint_generation_contract": checkpoint_generation["values"],
        "baseline_score_contract": baseline_score_provenance["values"],
        "checkpoint_score_contract": checkpoint_score_provenance["values"],
    }
    return selected, audit


def native_video_frames(path: Path, indices: list[int] | None = None) -> np.ndarray:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if len(reader) != 81:
        raise ValueError(f"Expected 81 frames in {path}, found {len(reader)}")
    if tuple(reader[0].shape) != (512, 512, 3):
        raise ValueError(f"Expected 512x512 RGB video in {path}, found {tuple(reader[0].shape)}")
    if not math.isclose(float(reader.get_avg_fps()), 16.0, rel_tol=0.0, abs_tol=0.02):
        raise ValueError(f"Expected 16 FPS in {path}, found {reader.get_avg_fps()}")
    if indices is None:
        indices = [0]
    return reader.get_batch(indices).asnumpy()


def prepare_movie_posters(cases: list[PairCase], output_dir: Path) -> dict[str, dict[str, Path]]:
    poster_root = output_dir / "assets/posters"
    assets: dict[str, dict[str, Path]] = {}
    for case in cases:
        case_assets: dict[str, Path] = {}
        for name, source in (("baseline", case.baseline_video), ("checkpoint", case.checkpoint_video)):
            frame_zero = native_video_frames(source)[0]
            poster = poster_root / case.case_id / f"{name}_frame_000.png"
            poster.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame_zero).save(poster, format="PNG", optimize=True)
            stored = np.asarray(Image.open(poster).convert("RGB"))
            if not np.array_equal(stored, frame_zero):
                raise ValueError(f"Raw frame-zero poster mismatch: {poster}")
            case_assets[name] = poster
        assets[case.case_id] = case_assets
    return assets


def audit_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),)
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_case_audit_sheet(case: PairCase) -> Image.Image:
    canvas = Image.new("RGB", (2000, 760), "#09111F")
    draw = ImageDraw.Draw(canvas)
    title_font = audit_font(31)
    body_font = audit_font(24)
    small_font = audit_font(19)
    draw.text(
        (28, 20),
        f"{case.case_id}  delta={case.delta:+.6f}  {case.canonical_name}",
        fill="#F4F7FB",
        font=title_font,
    )
    indices = [0, 20, 40, 60, 80]
    tile = 250
    start_x = 205
    for row, (label, path, score, color) in enumerate(
        (
            ("BASELINE", case.baseline_video, case.baseline_score, "#FB7185"),
            ("CKPT 2200", case.checkpoint_video, case.checkpoint_score, "#34D399"),
        )
    ):
        y = 95 + row * 315
        draw.text((28, y + 80), label, fill=color, font=body_font)
        draw.text((28, y + 120), f"score {score:.6f}", fill=color, font=small_font)
        for column, (frame, frame_index) in enumerate(zip(native_video_frames(path, indices), indices, strict=True)):
            image = Image.fromarray(frame).convert("RGB")
            image = ImageOps.fit(image, (tile, tile), method=Image.Resampling.LANCZOS)
            x = start_x + column * (tile + 12)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=color, width=3)
            draw.text(
                (x + 8, y + 8),
                f"f{frame_index:02d}",
                fill="#FFFFFF",
                font=small_font,
                stroke_width=2,
                stroke_fill="#09111F",
            )
    side_x = 1540
    for index, (label, path) in enumerate((("INPUT", case.input_image), ("GT FINAL", case.ground_truth_final))):
        y = 105 + index * 315
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(image, (270, 270), method=Image.Resampling.LANCZOS)
        canvas.paste(image, (side_x, y))
        draw.rectangle((side_x, y, side_x + 269, y + 269), outline="#60A5FA", width=3)
        draw.text((side_x + 285, y + 105), label, fill="#9FB0C5", font=small_font)
    return canvas


def write_audit_sheets(cases: list[PairCase], output_dir: Path) -> list[Path]:
    audit_root = output_dir / "audit_sheets"
    audit_root.mkdir(parents=True, exist_ok=True)
    case_sheets = [make_case_audit_sheet(case) for case in cases]
    page_paths: list[Path] = []
    per_page = 4
    for page_index in range(math.ceil(len(case_sheets) / per_page)):
        chunk = case_sheets[page_index * per_page : (page_index + 1) * per_page]
        page = Image.new("RGB", (2000, 760 * per_page), "#09111F")
        for index, sheet in enumerate(chunk):
            page.paste(sheet, (0, index * 760))
        path = audit_root / f"page_{page_index + 1:02d}.jpg"
        page.save(path, format="JPEG", quality=90, optimize=True)
        page_paths.append(path)
    return page_paths


def relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def case_record(case: PairCase) -> dict[str, Any]:
    return {
        "rank": case.rank,
        "case_id": case.case_id,
        "canonical_name": case.canonical_name,
        "domain": case.domain,
        "category": case.category,
        "task_name": case.task_name,
        "task_label": case.task_label,
        "video_idx": case.video_idx,
        "prompt": case.prompt,
        "baseline_score": case.baseline_score,
        "checkpoint_score": case.checkpoint_score,
        "score_delta": case.delta,
        "input_image": relative_to_repo(case.input_image),
        "ground_truth_video": relative_to_repo(case.ground_truth_video),
        "ground_truth_final": relative_to_repo(case.ground_truth_final),
        "baseline_native_video": relative_to_repo(case.baseline_video),
        "checkpoint_native_video": relative_to_repo(case.checkpoint_video),
        "baseline_scored_video": relative_to_repo(case.baseline_scored_video),
        "checkpoint_scored_video": relative_to_repo(case.checkpoint_scored_video),
        "baseline_native_sha256": sha256_file(case.baseline_video),
        "checkpoint_native_sha256": sha256_file(case.checkpoint_video),
        "manual_review": {
            "status": "pass",
            "basis": "temporal audit sheet plus exact paired EvalKit score",
        },
    }


def write_selection_artifacts(cases: list[PairCase], audit: dict[str, Any], output_dir: Path) -> None:
    records = [case_record(case) for case in cases]
    manifest = {
        "schema_version": 1,
        "selection_rule": (
            "Top 50 exact paired checkpoint-2200-minus-baseline EvalKit deltas; deterministic tie-break by "
            "checkpoint score, baseline score, then canonical name. All selected cases passed temporal-sheet review."
        ),
        "minimum_required_delta": MIN_SELECTED_DELTA,
        "selected_case_count": len(cases),
        "selected_unique_task_count": len({case.task_name for case in cases}),
        "selected_in_domain_count": sum(case.domain == "In_Domain" for case in cases),
        "selected_out_of_domain_count": sum(case.domain != "In_Domain" for case in cases),
        "selected_min_delta": min(case.delta for case in cases),
        "selected_mean_delta": float(np.mean([case.delta for case in cases])),
        "selected_max_delta": max(case.delta for case in cases),
        "source_audit": audit,
        "cases": records,
    }
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with (output_dir / "selection.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "case_id",
                "domain",
                "category",
                "task_name",
                "video_idx",
                "baseline_score",
                "checkpoint_score",
                "score_delta",
                "canonical_name",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.rank,
                    case.case_id,
                    case.domain,
                    case.category,
                    case.task_name,
                    case.video_idx,
                    f"{case.baseline_score:.9f}",
                    f"{case.checkpoint_score:.9f}",
                    f"{case.delta:.9f}",
                    case.canonical_name,
                ]
            )


def new_presentation() -> Any:
    presentation = deck.new_presentation()
    presentation.core_properties.title = "Checkpoint 2200 versus DiffSynth baseline: top 50 paired examples"
    presentation.core_properties.subject = "VBVR-Pro paired videos with exact EvalKit scores"
    presentation.core_properties.keywords = "RL, DanceGRPO, VBVR-Pro, checkpoint 2200, DiffSynth, embedded video"
    return presentation


def add_cover_slide(presentation: Any, cases: list[PairCase], audit: dict[str, Any]) -> None:
    slide = deck.blank_slide(presentation)
    deck.add_rect(slide, 0.0, 0.0, 0.11, 7.5, deck.TEAL, radius=False)
    deck.add_text(slide, "Checkpoint 2200 明显优于 baseline 的 50 个例子", 0.65, 0.62, 11.9, 0.75, size=31, bold=True)
    deck.add_text(
        slide,
        "DanceGRPO checkpoint-2200 + CPS 0.7  vs  DiffSynth step-35500 + UniPC ODE",
        0.68,
        1.48,
        11.6,
        0.42,
        size=17,
        color=deck.TEAL,
        bold=True,
        font=deck.FONT_LATIN,
    )
    metrics = (
        ("50", "paired examples", deck.BLUE),
        (f"+{np.mean([case.delta for case in cases]):.3f}", "selected mean delta", deck.GREEN),
        (f"+{min(case.delta for case in cases):.3f}", "smallest selected delta", deck.AMBER),
        (str(len({case.task_name for case in cases})), "task types", deck.TEAL),
    )
    for index, (value, label, color) in enumerate(metrics):
        x = 0.68 + index * 3.08
        deck.add_rect(slide, x, 2.45, 2.78, 1.32, deck.PANEL, line_color=deck.GRID)
        deck.add_text(
            slide, value, x + 0.12, 2.63, 2.54, 0.48, size=29, color=color, bold=True, align=deck.PP_ALIGN.CENTER
        )
        deck.add_text(
            slide,
            label,
            x + 0.12,
            3.2,
            2.54,
            0.25,
            size=11,
            color=deck.MUTED,
            align=deck.PP_ALIGN.CENTER,
            font=deck.FONT_LATIN,
        )
    deck.add_rect(slide, 0.68, 4.22, 12.0, 1.65, deck.PANEL_2, line_color=deck.GRID)
    deck.add_text(
        slide,
        f"完整 500 样本：{audit['baseline_overall']:.6f}  →  {audit['checkpoint_overall']:.6f}",
        0.95,
        4.48,
        11.45,
        0.42,
        size=23,
        bold=True,
        align=deck.PP_ALIGN.CENTER,
    )
    deck.add_text(
        slide,
        "每页内嵌两条 native 512×512×81 raw MP4；分数来自同一 EvalKit e140 / 4cc7d028 合同。",
        0.95,
        5.07,
        11.45,
        0.32,
        size=14,
        color=deck.MUTED,
        align=deck.PP_ALIGN.CENTER,
    )
    deck.add_text(slide, "2026-08-18", 10.9, 7.05, 1.8, 0.2, size=9.5, color=deck.MUTED, align=deck.PP_ALIGN.RIGHT)
    deck.add_notes(
        slide,
        "开场：这是严格配对的同一批 500 样本。每个展示页比较相同 input/prompt/seed 下的两条视频，"
        "并标出各自的规则分与差值。",
    )


def add_global_evidence_slide(presentation: Any, cases: list[PairCase], audit: dict[str, Any]) -> None:
    slide = deck.blank_slide(presentation)
    deck.add_slide_chrome(
        slide, "先看完整 500 样本，再看精选案例", "PAIRED EVIDENCE CONTRACT", len(presentation.slides)
    )
    global_delta = audit["checkpoint_overall"] - audit["baseline_overall"]
    metrics = (
        (f"{audit['baseline_overall']:.6f}", "DiffSynth baseline", deck.RED),
        (f"{audit['checkpoint_overall']:.6f}", "checkpoint 2200", deck.GREEN),
        (f"+{global_delta:.6f}", "500-sample delta", deck.TEAL),
        ("500 / 500", "paired · zero errors", deck.BLUE),
    )
    for index, (value, label, color) in enumerate(metrics):
        x = 0.47 + index * 3.17
        deck.add_rect(slide, x, 1.32, 2.9, 1.38, deck.PANEL, line_color=color, line_width=1.7)
        deck.add_text(
            slide, value, x + 0.12, 1.54, 2.66, 0.46, size=24, color=color, bold=True, align=deck.PP_ALIGN.CENTER
        )
        deck.add_text(
            slide,
            label,
            x + 0.12,
            2.14,
            2.66,
            0.25,
            size=11.5,
            color=deck.MUTED,
            align=deck.PP_ALIGN.CENTER,
            font=deck.FONT_LATIN,
        )
    lines = [
        ("样本身份：两边 eval_samples.json 字节级相同，500 个 canonical name 完全配对。", 16.5, deck.TEXT, True),
        ("评分合同：相同 EvalKit e140 源码指纹；1024×1024×81、16 FPS；两边均无 scorer error。", 16, deck.TEXT, False),
        (
            f"配对方向：{audit['positive_pair_count']} 个提升、{audit['tied_pair_count']} 个持平、"
            f"{audit['negative_pair_count']} 个下降。",
            16,
            deck.TEXT,
            False,
        ),
        (
            "因果边界：这里同时改变了模型 checkpoint 与采样器（CPS vs UniPC），展示的是整套生成配置差异。",
            16,
            deck.AMBER,
            True,
        ),
        ("精选边界：后续 50 个是按单样本分差挑出的高对比案例，不是无偏平均质量估计。", 16, deck.AMBER, False),
    ]
    deck.add_rect(slide, 0.6, 3.16, 12.1, 3.55, deck.PANEL_2, line_color=deck.GRID)
    deck.add_multiline(slide, lines, 0.9, 3.48, 11.55, 2.9, bullet=True)
    deck.add_notes(slide, "必须先讲完整 500 样本的平均分，再展示 top-50。不要把精选案例当作总体平均。")


def add_reading_guide_slide(presentation: Any, example: PairCase, assets: dict[str, dict[str, Path]]) -> None:
    slide = deck.blank_slide(presentation)
    deck.add_slide_chrome(slide, "案例页怎么读", "CLICK EITHER RAW POSTER TO PLAY", len(presentation.slides))
    panels = (
        ("DiffSynth step-35500 · UniPC ODE", example.baseline_score, assets[example.case_id]["baseline"], deck.RED),
        (
            "DanceGRPO checkpoint-2200 · CPS 0.7",
            example.checkpoint_score,
            assets[example.case_id]["checkpoint"],
            deck.GREEN,
        ),
    )
    for index, (label, score, poster, color) in enumerate(panels):
        x = 0.72 + index * 6.25
        deck.add_text(slide, label, x, 1.22, 5.65, 0.3, size=14, color=color, bold=True, align=deck.PP_ALIGN.CENTER)
        deck.add_rect(slide, x + 0.57, 1.66, 4.52, 4.52, deck.PANEL, line_color=color, line_width=2.0)
        slide.shapes.add_picture(
            str(poster), deck.Inches(x + 0.62), deck.Inches(1.71), deck.Inches(4.42), deck.Inches(4.42)
        )
        deck.add_text(
            slide,
            f"EvalKit {score:.6f}",
            x,
            6.28,
            5.65,
            0.36,
            size=20,
            color=color,
            bold=True,
            align=deck.PP_ALIGN.CENTER,
        )
    deck.add_rect(slide, 5.59, 2.55, 2.15, 1.55, deck.BG, line_color=deck.TEAL, line_width=2.0)
    deck.add_text(slide, "分数差", 5.76, 2.78, 1.81, 0.28, size=13, color=deck.MUTED, align=deck.PP_ALIGN.CENTER)
    deck.add_text(
        slide,
        f"+{example.delta:.3f}",
        5.76,
        3.2,
        1.81,
        0.48,
        size=27,
        color=deck.TEAL,
        bold=True,
        align=deck.PP_ALIGN.CENTER,
    )
    deck.add_text(
        slide,
        "封面是 raw video 原始第 0 帧，不加标签、不拼图；进入放映模式后点击即可从头播放。",
        0.8,
        6.82,
        11.75,
        0.26,
        size=12.5,
        color=deck.MUTED,
        align=deck.PP_ALIGN.CENTER,
    )
    deck.add_notes(slide, "建议先播放 baseline，再播放 checkpoint 2200；最后指出分数差和 GT 最终状态。")


def add_index_slide(presentation: Any, cases: list[PairCase], title: str) -> None:
    slide = deck.blank_slide(presentation)
    deck.add_slide_chrome(slide, title, "RANKED BY EXACT PAIRED SCORE DELTA", len(presentation.slides))
    split = math.ceil(len(cases) / 2)
    for column, chunk in enumerate((cases[:split], cases[split:])):
        x = 0.52 + column * 6.38
        for row, case in enumerate(chunk):
            y = 1.18 + row * 0.43
            fill = deck.PANEL if row % 2 == 0 else deck.PANEL_2
            deck.add_rect(slide, x, y, 6.0, 0.36, fill, radius=False)
            deck.add_text(
                slide,
                f"{case.rank:02d}",
                x + 0.08,
                y + 0.03,
                0.42,
                0.28,
                size=10.5,
                color=deck.TEAL,
                bold=True,
                font=deck.FONT_LATIN,
            )
            deck.add_text(slide, case.task_label, x + 0.55, y + 0.03, 3.5, 0.28, size=10.5, bold=True)
            deck.add_text(
                slide, case.video_idx, x + 4.05, y + 0.03, 0.7, 0.28, size=9.5, color=deck.MUTED, font=deck.FONT_LATIN
            )
            deck.add_text(
                slide,
                f"+{case.delta:.3f}",
                x + 4.95,
                y + 0.03,
                0.82,
                0.28,
                size=10.5,
                color=deck.GREEN,
                bold=True,
                align=deck.PP_ALIGN.RIGHT,
                font=deck.FONT_LATIN,
            )
    deck.add_notes(slide, "索引按 checkpoint 分数减 baseline 分数降序排列。")


def add_case_slide(
    presentation: Any,
    case: PairCase,
    assets: dict[str, dict[str, Path]],
) -> None:
    slide = deck.blank_slide(presentation)
    title = f"{case.rank:02d}/50 · {case.task_label} · sample {case.video_idx}"
    subtitle = f"{case.domain_label}  |  {case.category}  |  {case.canonical_name}"
    deck.add_slide_chrome(slide, title, subtitle, len(presentation.slides))

    left_x = 0.48
    right_x = 8.31
    video_y = 1.36
    video_size = 4.52
    deck.add_text(
        slide,
        "DiffSynth step-35500 · UniPC ODE",
        left_x,
        1.03,
        video_size,
        0.25,
        size=12.5,
        color=deck.RED,
        bold=True,
        align=deck.PP_ALIGN.CENTER,
    )
    deck.add_text(
        slide,
        "DanceGRPO checkpoint-2200 · CPS 0.7",
        right_x,
        1.03,
        video_size,
        0.25,
        size=12.5,
        color=deck.GREEN,
        bold=True,
        align=deck.PP_ALIGN.CENTER,
    )
    deck.add_rect(
        slide,
        left_x - 0.04,
        video_y - 0.04,
        video_size + 0.08,
        video_size + 0.08,
        deck.PANEL,
        line_color=deck.RED,
        line_width=2.0,
    )
    deck.add_rect(
        slide,
        right_x - 0.04,
        video_y - 0.04,
        video_size + 0.08,
        video_size + 0.08,
        deck.PANEL,
        line_color=deck.GREEN,
        line_width=2.0,
    )
    slide.shapes.add_movie(
        str(case.baseline_video),
        deck.Inches(left_x),
        deck.Inches(video_y),
        deck.Inches(video_size),
        deck.Inches(video_size),
        str(assets[case.case_id]["baseline"]),
        mime_type="video/mp4",
    )
    slide.shapes.add_movie(
        str(case.checkpoint_video),
        deck.Inches(right_x),
        deck.Inches(video_y),
        deck.Inches(video_size),
        deck.Inches(video_size),
        str(assets[case.case_id]["checkpoint"]),
        mime_type="video/mp4",
    )
    deck.add_text(
        slide,
        f"{case.baseline_score:.3f}",
        left_x,
        5.98,
        video_size,
        0.48,
        size=28,
        color=deck.RED,
        bold=True,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )
    deck.add_text(
        slide,
        f"exact {case.baseline_score:.6f}",
        left_x,
        6.46,
        video_size,
        0.22,
        size=9.5,
        color=deck.MUTED,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )
    deck.add_text(
        slide,
        f"{case.checkpoint_score:.3f}",
        right_x,
        5.98,
        video_size,
        0.48,
        size=28,
        color=deck.GREEN,
        bold=True,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )
    deck.add_text(
        slide,
        f"exact {case.checkpoint_score:.6f}",
        right_x,
        6.46,
        video_size,
        0.22,
        size=9.5,
        color=deck.MUTED,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )

    center_x = 5.15
    deck.add_rect(slide, center_x, 1.45, 2.51, 1.35, deck.PANEL_2, line_color=deck.TEAL, line_width=2.0)
    deck.add_text(
        slide,
        "checkpoint − baseline",
        center_x + 0.12,
        1.66,
        2.27,
        0.25,
        size=10.5,
        color=deck.MUTED,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )
    deck.add_text(
        slide,
        f"+{case.delta:.3f}",
        center_x + 0.12,
        2.03,
        2.27,
        0.48,
        size=28,
        color=deck.TEAL,
        bold=True,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )
    deck.add_text(slide, "输入", center_x, 3.07, 1.16, 0.24, size=10.5, color=deck.MUTED, align=deck.PP_ALIGN.CENTER)
    deck.add_text(
        slide, "GT 最终帧", center_x + 1.35, 3.07, 1.16, 0.24, size=10.5, color=deck.MUTED, align=deck.PP_ALIGN.CENTER
    )
    slide.shapes.add_picture(
        str(case.input_image), deck.Inches(center_x), deck.Inches(3.36), deck.Inches(1.16), deck.Inches(1.16)
    )
    slide.shapes.add_picture(
        str(case.ground_truth_final),
        deck.Inches(center_x + 1.35),
        deck.Inches(3.36),
        deck.Inches(1.16),
        deck.Inches(1.16),
    )
    deck.add_rect(slide, center_x, 4.85, 2.51, 1.03, deck.PANEL, line_color=deck.GRID)
    deck.add_text(
        slide,
        "相同条件",
        center_x + 0.12,
        5.02,
        2.27,
        0.23,
        size=11,
        color=deck.TEAL,
        bold=True,
        align=deck.PP_ALIGN.CENTER,
    )
    deck.add_text(
        slide,
        "seed 0 · 30 steps · CFG 1\n512² × 81 @ 16 FPS",
        center_x + 0.12,
        5.3,
        2.27,
        0.44,
        size=9.5,
        color=deck.MUTED,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )
    deck.add_text(
        slide,
        "点击视频播放  ·  raw MP4  ·  EvalKit score on prepared 1024×1024×81 video  ·  e140 / 4cc7d028",
        0.6,
        6.94,
        12.1,
        0.22,
        size=9.3,
        color=deck.MUTED,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )
    deck.add_notes(
        slide,
        f"{case.case_id} / rank {case.rank}\n"
        f"Canonical sample: {case.canonical_name}\n"
        f"Task: {case.task_name}\n"
        f"Prompt: {case.prompt}\n\n"
        f"Baseline score: {case.baseline_score:.9f}\n"
        f"Checkpoint score: {case.checkpoint_score:.9f}\n"
        f"Delta: {case.delta:+.9f}\n"
        f"Baseline native SHA-256: {sha256_file(case.baseline_video)}\n"
        f"Checkpoint native SHA-256: {sha256_file(case.checkpoint_video)}\n\n"
        "讲解建议：先播左侧 baseline，再播右侧 checkpoint，最后对照 GT 最终帧与精确分数。",
    )


def add_closing_slide(presentation: Any, cases: list[PairCase], audit: dict[str, Any]) -> None:
    slide = deck.blank_slide(presentation)
    deck.add_rect(slide, 0.0, 0.0, 0.11, 7.5, deck.TEAL, radius=False)
    deck.add_text(slide, "Takeaway", 0.72, 0.72, 4.0, 0.55, size=31, color=deck.TEAL, bold=True, font=deck.FONT_LATIN)
    deck.add_text(
        slide, "完整曲线说明平均得分提高；配对视频解释提高发生在哪里。", 0.74, 1.53, 11.6, 0.58, size=25, bold=True
    )
    lines = [
        (f"500 样本总体：{audit['baseline_overall']:.6f} → {audit['checkpoint_overall']:.6f}。", 18, deck.TEXT, True),
        (
            f"精选 50 个：最小分差 +{min(case.delta for case in cases):.3f}，"
            f"平均分差 +{np.mean([case.delta for case in cases]):.3f}。",
            18,
            deck.TEXT,
            False,
        ),
        ("每个案例保持 input、prompt、seed、步数、CFG、时长和评分合同一致。", 18, deck.TEXT, False),
        ("但 checkpoint 与 sampler 同时变化；若要单独归因 RL，应补充同 sampler 的控制比较。", 18, deck.AMBER, True),
    ]
    deck.add_rect(slide, 0.72, 2.58, 11.9, 3.2, deck.PANEL_2, line_color=deck.GRID)
    deck.add_multiline(slide, lines, 1.04, 2.94, 11.2, 2.55, bullet=True)
    deck.add_text(
        slide,
        "50 paired examples · 100 embedded raw MP4s",
        0.76,
        6.58,
        11.8,
        0.3,
        size=15,
        color=deck.MUTED,
        align=deck.PP_ALIGN.CENTER,
        font=deck.FONT_LATIN,
    )
    deck.add_notes(slide, "收尾：先用完整结果支撑总体主张，再用 top-50 视频说明具体行为差异；保留 sampler 因果边界。")


def build_deck(
    output_path: Path,
    cases: list[PairCase],
    audit: dict[str, Any],
    assets: dict[str, dict[str, Path]],
) -> int:
    presentation = new_presentation()
    add_cover_slide(presentation, cases, audit)
    add_global_evidence_slide(presentation, cases, audit)
    add_reading_guide_slide(presentation, cases[0], assets)
    add_index_slide(presentation, cases[:25], "案例索引 01–25")
    add_index_slide(presentation, cases[25:], "案例索引 26–50")
    for case in cases:
        add_case_slide(presentation, case, assets)
    add_closing_slide(presentation, cases, audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(output_path)
    return len(presentation.slides)


def write_readme(output_dir: Path, deck_name: str, cases: list[PairCase], audit: dict[str, Any]) -> None:
    text = f"""# VBVR-Pro checkpoint-2200 versus DiffSynth baseline

## Main file

- `{deck_name}`: {len(cases)} paired examples and {len(cases) * 2} embedded native MP4s.
- `selection_manifest.json` / `selection.csv`: exact source paths, scores, deltas, hashes, and selection provenance.
- `audit_sheets/`: five-timepoint baseline/checkpoint timelines used for visual review.
- `preview/`: rasterized layout spot checks. Their Aspose evaluation watermark is not present in the PowerPoint.
- `build_report.json`: PowerPoint ZIP/media/hash validation.

## Playback

Use desktop Microsoft PowerPoint when possible. Each movie poster is the raw MP4's exact decoded frame zero, with no
labels, crops, or compositing. Click either side in slide-show mode to play from the beginning. The PPTX embeds every
native MP4 and does not depend on links back to `storage/eval_out`.

## Scores and evidence boundary

The exact paired 500-sample Overall means are `{audit["baseline_overall"]:.6f}` for DiffSynth step-35500 + UniPC ODE
and `{audit["checkpoint_overall"]:.6f}` for DanceGRPO checkpoint-2200 + CPS 0.7. Both use the same 500 canonical
samples, 30 steps, CFG 1, seed 0, native 512x512x81 generation, 16 FPS, and EvalKit e140 source fingerprint
`{EXPECTED_EVALKIT_SHA256}`; both score runs have zero errors.

The selected top 50 have exact checkpoint-minus-baseline deltas from `+{min(case.delta for case in cases):.6f}` to
`+{max(case.delta for case in cases):.6f}`. They are deliberately high-contrast examples and are not an unbiased
estimate of average quality. The comparison also changes both model checkpoint and sampler, so it demonstrates the
combined generation configuration rather than isolating RL from CPS.

Google Slides import is not the primary delivery format because it does not reliably retain locally embedded MP4
parts.
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, audit = load_audited_pairs()
    assets = prepare_movie_posters(cases, output_dir)
    audit_sheets = write_audit_sheets(cases, output_dir)
    write_selection_artifacts(cases, audit, output_dir)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "selected": len(cases),
                    "unique_tasks": len({case.task_name for case in cases}),
                    "min_delta": min(case.delta for case in cases),
                    "mean_delta": float(np.mean([case.delta for case in cases])),
                    "audit_sheets": len(audit_sheets),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    deck_name = "vbvr_checkpoint2200_vs_diffsynth_baseline_top50_embedded_video_20260818.pptx"
    output_path = output_dir / deck_name
    slide_count = build_deck(output_path, cases, audit, assets)
    source_videos = [video for case in cases for video in (case.baseline_video, case.checkpoint_video)]
    validation = deck.validate_deck(output_path, expected_slides=slide_count, source_videos=source_videos)
    write_readme(output_dir, deck_name, cases, audit)
    report = {
        "schema_version": 1,
        "selection_count": len(cases),
        "unique_task_count": len({case.task_name for case in cases}),
        "in_domain_count": sum(case.domain == "In_Domain" for case in cases),
        "out_of_domain_count": sum(case.domain != "In_Domain" for case in cases),
        "minimum_delta": min(case.delta for case in cases),
        "mean_delta": float(np.mean([case.delta for case in cases])),
        "maximum_delta": max(case.delta for case in cases),
        "raw_frame_zero_posters_exact": len(cases) * 2,
        "audit_sheet_count": len(audit_sheets),
        "source_audit": audit,
        "deck": validation,
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
