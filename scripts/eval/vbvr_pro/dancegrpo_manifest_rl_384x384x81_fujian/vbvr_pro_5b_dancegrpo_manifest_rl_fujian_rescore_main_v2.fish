#!/usr/bin/env fish

# Re-score already prepared 500-video outputs from the Fujian manifest-RL
# sweep. This intentionally does not load checkpoints, generate videos, or run
# video preparation. The native generation dimensions are checked through the
# recorded generation provenance before EvalKit starts. New score artifacts
# and their provenance are written under a separate output namespace.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -g PYTHON .venv/bin/python
set -g RESCORE_SCRIPT (realpath (status filename))

set -q SOURCE_OUTPUT_BASE[1]
or set -gx SOURCE_OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_384x384x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_evalkit_eb977da6
set -q OUTPUT_BASE[1]
or set -gx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_384x384x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_rescore_from_evalkit_eb977da6_to_evalkit_4cc7d028
set -q EVAL_LOG_DIR[1]
or set -gx EVAL_LOG_DIR storage/eval_logs/vbvr_pro_main_v2_384x384x81_manifest_rl_fujian_rescore_evalkit_4cc7d028

set -q GT_BASE[1]
or set -gx GT_BASE (realpath storage/datasets/vbvr-pro-eval-500)
set -q EVALKIT_DIR[1]
or set -gx EVALKIT_DIR storage/evalkits/vbvr-evalkit-interleave-main_v2-e140038f
set -q EVALKIT_REV[1]
or set -gx EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -q EVALKIT_SOURCE_SHA256[1]
or set -gx EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
set -q EASYOCR_ROOT[1]
or set -gx EASYOCR_ROOT storage/evalkits/easyocr-shared

set -q EXPECTED_VIDEOS[1]; or set -gx EXPECTED_VIDEOS 500
set -q EXPECTED_TASKS[1]; or set -gx EXPECTED_TASKS 100
set -q SCORE_WORKERS[1]; or set -gx SCORE_WORKERS 2
set -q SCORE_THREADS_PER_WORKER[1]; or set -gx SCORE_THREADS_PER_WORKER 8
set -q MAX_PARALLEL[1]; or set -gx MAX_PARALLEL 4
set -q EXPECTED_GENERATION_WIDTH[1]; or set -gx EXPECTED_GENERATION_WIDTH 384
set -q EXPECTED_GENERATION_HEIGHT[1]; or set -gx EXPECTED_GENERATION_HEIGHT 384
set -g PREPARED_WIDTH 1024
set -g PREPARED_HEIGHT 1024
set -g PREPARED_NAME eval_1024x1024_81f_fps16_5p0625s

test -d $SOURCE_OUTPUT_BASE; or _fail "source output base does not exist: $SOURCE_OUTPUT_BASE"
test -d $GT_BASE; or _fail "GT_BASE does not exist: $GT_BASE"
test -f $GT_BASE/SHA256SUMS; or _fail "dataset checksum manifest is missing: $GT_BASE/SHA256SUMS"
test ! -e $GT_BASE/.cache
or _fail "move the Hugging Face local-dir .cache outside GT_BASE before provenance fingerprinting: $GT_BASE/.cache"
test -f $EVALKIT_DIR/run_evaluation.py; or _fail "EvalKit checkout is incomplete: $EVALKIT_DIR"
test -f $EASYOCR_ROOT/model/craft_mlt_25k.pth; or _fail "EasyOCR CRAFT weight is missing"
test -f $EASYOCR_ROOT/model/english_g2.pth; or _fail "EasyOCR English weight is missing"

for value_name in EXPECTED_VIDEOS EXPECTED_TASKS SCORE_WORKERS SCORE_THREADS_PER_WORKER MAX_PARALLEL EXPECTED_GENERATION_WIDTH EXPECTED_GENERATION_HEIGHT PREPARED_WIDTH PREPARED_HEIGHT
    set -l value $$value_name
    string match -rq '^[1-9][0-9]*$' -- $value
    or _fail "$value_name must be a positive integer, got $value"
end

set -l native_thread_budget (math $SCORE_WORKERS \* $SCORE_THREADS_PER_WORKER \* $MAX_PARALLEL)
set -l available_cpus (nproc)
test $native_thread_budget -le $available_cpus
or _fail "scorer native-thread budget $native_thread_budget exceeds nproc=$available_cpus"

set -l expected_checksums_sha256 a67c534293724ddfc6657af755ab65e9b1354879deb2cfc47de22ede43942861
set -l actual_checksums_sha256 (sha256sum $GT_BASE/SHA256SUMS | awk '{print $1}')
test "$actual_checksums_sha256" = "$expected_checksums_sha256"
or _fail "dataset checksum-manifest mismatch: expected=$expected_checksums_sha256 actual=$actual_checksums_sha256"

if not set -q WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED[1]
    echo "[dataset] verifying the complete downloaded VBVR-Pro eval snapshot"
    pushd $GT_BASE >/dev/null; or exit 1
    sha256sum -c SHA256SUMS --quiet
    set -l checksum_status $status
    popd >/dev/null; or exit 1
    test $checksum_status -eq 0; or _fail "VBVR-Pro eval snapshot failed SHA-256 verification"
end
set -gx WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED 1

set -g evalkit_revision_actual (git -C $EVALKIT_DIR rev-parse HEAD)
or _fail "could not resolve EvalKit revision: $EVALKIT_DIR"
test "$evalkit_revision_actual" = "$EVALKIT_REV"
or _fail "EvalKit revision mismatch: expected=$EVALKIT_REV actual=$evalkit_revision_actual"

set -g evalkit_source_sha256 ($PYTHON -c '
import sys
from src.eval.vbvr_run_evaluation_parallel import evalkit_source_sha256
print(evalkit_source_sha256(sys.argv[1]))
' $EVALKIT_DIR)
or _fail "could not fingerprint EvalKit source"
test "$evalkit_source_sha256" = "$EVALKIT_SOURCE_SHA256"
or _fail "EvalKit source fingerprint mismatch: expected=$EVALKIT_SOURCE_SHA256 actual=$evalkit_source_sha256"

set -l evalkit_easyocr_models $EVALKIT_DIR/easyocr_models
set -l expected_easyocr_models (realpath $EASYOCR_ROOT/model)
test -L $evalkit_easyocr_models
or _fail "EvalKit EasyOCR model link is missing: $evalkit_easyocr_models"
test (realpath $evalkit_easyocr_models) = $expected_easyocr_models
or _fail "EvalKit EasyOCR model link points to the wrong directory: $evalkit_easyocr_models"

set -g scorer_dependency_versions (env \
    OMP_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    MKL_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    OPENBLAS_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    NUMEXPR_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    $PYTHON -m src.eval.vbvr_runtime --json
)
or _fail "main_v2 scorer runtime contract failed; run uv sync --frozen and restart"
test -n "$scorer_dependency_versions"
or _fail "main_v2 scorer dependency record is empty"

function _source_root_for_step
    echo $SOURCE_OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$argv[1]-cps-noise-0.7
end

function _output_root_for_step
    echo $OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$argv[1]-cps-noise-0.7
end

function _source_valid
    set -l step $argv[1]
    set -l source_root (_source_root_for_step $step)
    set -l prepared_dir $source_root/$PREPARED_NAME
    set -l preparation_provenance $source_root/preparation-provenance.json
    $PYTHON -c '
import json
import sys
from pathlib import Path

from src.eval.evaluation_provenance import verify_recorded_manifest

prepared = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
expected_videos = int(sys.argv[3])
expected_width = sys.argv[4]
expected_height = sys.argv[5]
expected_generation_width = sys.argv[6]
expected_generation_height = sys.argv[7]
matches, detail = verify_recorded_manifest(
    manifest_path,
    expected_stage="vbvr-pro-preparation",
    require_complete=True,
)
if not matches:
    raise SystemExit(detail)
manifest = json.loads(manifest_path.read_text())
values = manifest.get("values", {})
for key, expected in (("width", expected_width), ("height", expected_height)):
    if values.get(key) != expected:
        raise SystemExit(
            f"prepared-video provenance has {key}={values.get(key)!r}, expected={expected!r}"
        )
generation_manifest = (manifest_path.parent / "generation-provenance.json").resolve()
generation_record = manifest.get("files", {}).get("generation_provenance", {})
if generation_record.get("path") != str(generation_manifest):
    raise SystemExit(
        "preparation provenance is not bound to the source run generation manifest: "
        f"{generation_record.get('path')!r} != {str(generation_manifest)!r}"
    )
generation = json.loads(generation_manifest.read_text())
if generation.get("stage") != "vbvr-pro-generation":
    raise SystemExit("source generation provenance has the wrong stage")
generation_values = generation.get("values", {})
expected_generation_values = {
    "state": "complete",
    "width": expected_generation_width,
    "height": expected_generation_height,
    "num_frames": "81",
    "fps": "16",
    "generation_mode": "cps",
    "num_inference_steps": "30",
    "guidance_scale": "1.0",
    "cps_noise_level": "0.7",
    "seed": "0",
}
for key, expected in expected_generation_values.items():
    if generation_values.get(key) != expected:
        raise SystemExit(
            f"source generation provenance has {key}={generation_values.get(key)!r}, expected={expected!r}"
        )
if generation.get("media_trees", {}).get("generated_videos", {}).get("entries") != expected_videos:
    raise SystemExit("source generation provenance does not contain the expected video count")
recorded = manifest.get("media_trees", {}).get("prepared_videos", {})
if recorded.get("path") != str(prepared):
    raise SystemExit(f"prepared-video path mismatch: {recorded.get('path')!r} != {str(prepared)!r}")
if recorded.get("entries") != expected_videos:
    raise SystemExit(f"prepared-video provenance has {recorded.get('entries')} entries")
count = sum(1 for _ in prepared.rglob("*.mp4"))
if count != expected_videos:
    raise SystemExit(f"prepared-video tree has {count}/{expected_videos} MP4s")
' $prepared_dir $preparation_provenance $EXPECTED_VIDEOS $PREPARED_WIDTH $PREPARED_HEIGHT \
        $EXPECTED_GENERATION_WIDTH $EXPECTED_GENERATION_HEIGHT
end

function _valid_score_result
    set -l result_path $argv[1]
    set -l prepared_dir $argv[2]
    $PYTHON -c '
import json
import math
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
prepared = Path(sys.argv[2]).resolve()
expected_videos = int(sys.argv[3])
expected_tasks = int(sys.argv[4])
data = json.loads(result_path.read_text())
samples = data.get("samples")
if not isinstance(samples, list) or len(samples) != expected_videos:
    raise SystemExit("result does not contain the expected sample count")
if any(sample.get("error") for sample in samples):
    raise SystemExit("result contains scorer errors")
if any(not math.isfinite(float(sample.get("score"))) for sample in samples):
    raise SystemExit("result contains a non-finite score")
if len({sample.get("task_name") for sample in samples}) != expected_tasks:
    raise SystemExit("result does not contain the expected task count")
for sample in samples:
    video_path = Path(sample.get("video_path", "")).resolve()
    if not video_path.is_relative_to(prepared):
        raise SystemExit(f"result points outside the prepared source: {video_path}")
summary = data.get("summary", {})
for key, count in (("In_Domain", 250), ("Out_of_Domain", 250), ("overall", expected_videos)):
    if summary.get(key, {}).get("num_samples") != count:
        raise SystemExit(f"invalid {key} summary count")
if len(summary.get("overall", {}).get("by_task", {})) != expected_tasks:
    raise SystemExit("invalid overall task summary")
' $result_path $prepared_dir $EXPECTED_VIDEOS $EXPECTED_TASKS
end

function _score_provenance
    set -l mode $argv[1]
    set -l state $argv[2]
    set -l result_path $argv[3]
    set -l prepared_dir $argv[4]
    set -l source_preparation_provenance $argv[5]
    set -l score_provenance $argv[6]
    set -l output_args
    if test $mode = promote
        set output_args --output-file result=$result_path
    else if test $mode = check; and test $state = complete
        set output_args --output-file result=$result_path
    end
    $PYTHON -m src.eval.evaluation_provenance $mode \
        --manifest $score_provenance \
        --stage vbvr-pro-score \
        --value state=$state \
        --value scoring_mode=rescore_existing_prepared_videos \
        --value expected_videos=$EXPECTED_VIDEOS \
        --value expected_tasks=$EXPECTED_TASKS \
        --value evalkit_revision=$EVALKIT_REV \
        --value evalkit_revision_actual=$evalkit_revision_actual \
        --value evalkit_source_sha256=$evalkit_source_sha256 \
        --value scorer_dependencies=$scorer_dependency_versions \
        --value device=cpu \
        --value workers=$SCORE_WORKERS \
        --value threads_per_worker=$SCORE_THREADS_PER_WORKER \
        --file source_preparation_provenance=$source_preparation_provenance \
        --file scorer_entrypoint=$EVALKIT_DIR/run_evaluation.py \
        --file scorer_requirements=$EVALKIT_DIR/requirements.txt \
        --file scorer_runtime=src/eval/vbvr_runtime.py \
        --file easyocr_craft=$EASYOCR_ROOT/model/craft_mlt_25k.pth \
        --file easyocr_english=$EASYOCR_ROOT/model/english_g2.pth \
        --file scorer_wrapper=src/eval/vbvr_run_evaluation_parallel.py \
        --tree prepared_videos=$prepared_dir \
        --tree ground_truth=$GT_BASE \
        $output_args
end

function _checkpoint_complete
    set -l step $argv[1]
    set -l source_root (_source_root_for_step $step)
    set -l output_root (_output_root_for_step $step)
    set -l prepared_dir $source_root/$PREPARED_NAME
    set -l source_preparation_provenance $source_root/preparation-provenance.json
    set -l score_provenance $output_root/score-provenance.json
    set -l result_path $output_root/scores/$PREPARED_NAME"_vbvr_results.json"
    set -l workbook $output_root/scores/$PREPARED_NAME"_task_scores.xlsx"
    set -l summary_path $output_root/final_scores.txt

    test -f $result_path; and test -f $score_provenance
    and test -f $workbook; and test -f $summary_path
    or return 1
    _source_valid $step; or return 1
    _valid_score_result $result_path $prepared_dir; or return 1

    $PYTHON -c '
import json
import sys
from pathlib import Path

from src.eval.evaluation_provenance import verify_recorded_manifest

manifest_path = Path(sys.argv[1]).resolve()
result = Path(sys.argv[2]).resolve()
prepared = Path(sys.argv[3]).resolve()
ground_truth = Path(sys.argv[4]).resolve()
source_preparation = Path(sys.argv[5]).resolve()
expected_revision = sys.argv[6]
expected_source_sha256 = sys.argv[7]
expected_runtime = json.loads(sys.argv[8])
expected_videos = sys.argv[9]
expected_tasks = sys.argv[10]

matches, detail = verify_recorded_manifest(
    manifest_path,
    expected_stage="vbvr-pro-score",
    require_complete=True,
)
if not matches:
    raise SystemExit(detail)
manifest = json.loads(manifest_path.read_text())
values = manifest.get("values", {})
expected_values = {
    "state": "complete",
    "scoring_mode": "rescore_existing_prepared_videos",
    "expected_videos": expected_videos,
    "expected_tasks": expected_tasks,
    "evalkit_revision": expected_revision,
    "evalkit_revision_actual": expected_revision,
    "evalkit_source_sha256": expected_source_sha256,
    "device": "cpu",
}
for key, expected in expected_values.items():
    if values.get(key) != expected:
        raise SystemExit(f"score provenance value mismatch for {key}")
if json.loads(values.get("scorer_dependencies", "null")) != expected_runtime:
    raise SystemExit("score provenance runtime contract mismatch")
bindings = (
    ("files", "source_preparation_provenance", source_preparation),
    ("trees", "prepared_videos", prepared),
    ("trees", "ground_truth", ground_truth),
    ("output_files", "result", result),
)
for section, name, expected in bindings:
    actual = manifest.get(section, {}).get(name, {}).get("path")
    if actual != str(expected):
        raise SystemExit(f"score provenance path mismatch for {section}.{name}")
' $score_provenance $result_path $prepared_dir $GT_BASE $source_preparation_provenance \
        $EVALKIT_REV $EVALKIT_SOURCE_SHA256 $scorer_dependency_versions $EXPECTED_VIDEOS $EXPECTED_TASKS
end

function _run_step
    set -l step $argv[1]
    set -l source_root (_source_root_for_step $step)
    set -l output_root (_output_root_for_step $step)
    set -l prepared_dir $source_root/$PREPARED_NAME
    set -l source_preparation_provenance $source_root/preparation-provenance.json
    set -l score_dir $output_root/scores
    set -l score_provenance $output_root/score-provenance.json
    set -l result_path $score_dir/$PREPARED_NAME"_vbvr_results.json"
    set -l workbook $score_dir/$PREPARED_NAME"_task_scores.xlsx"
    set -l summary_path $output_root/final_scores.txt

    if _checkpoint_complete $step
        echo "[skip] checkpoint-$step scorer-only result is already complete"
        return 0
    end
    _source_valid $step
    or begin
        echo "[error] checkpoint-$step source prepared-video validation failed" >&2
        return 1
    end

    if set -q DRY_RUN[1]
        echo "[dry-run] checkpoint-$step source=$prepared_dir output=$output_root"
        return 0
    end

    mkdir -p $score_dir; or return 1
    rm -f -- $result_path $workbook $summary_path $score_provenance
    _score_provenance write in_progress_rewrite $result_path $prepared_dir \
        $source_preparation_provenance $score_provenance
    or return 1

    echo "[score] checkpoint-$step e140 EvalKit only; source=$prepared_dir"
    env CUDA_VISIBLE_DEVICES= EASYOCR_MODULE_PATH=(realpath $EASYOCR_ROOT) \
        OMP_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
        MKL_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
        OPENBLAS_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
        NUMEXPR_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
        $PYTHON -m src.eval.vbvr_run_evaluation_parallel \
        --model_path $prepared_dir \
        --gt_base $GT_BASE \
        --output_dir $score_dir \
        --evalkit_dir $EVALKIT_DIR \
        --expected_evalkit_source_sha256 $EVALKIT_SOURCE_SHA256 \
        --expected_videos $EXPECTED_VIDEOS \
        --device cpu \
        --num_workers $SCORE_WORKERS \
        --threads_per_worker $SCORE_THREADS_PER_WORKER
    or return 1

    _valid_score_result $result_path $prepared_dir; or return 1
    $PYTHON -m src.cli.export_vbvr_task_scores $result_path \
        --output $workbook \
        --summary-output $summary_path \
        --expected-samples $EXPECTED_VIDEOS \
        --expected-tasks $EXPECTED_TASKS
    or return 1
    _score_provenance promote in_progress_rewrite $result_path $prepared_dir \
        $source_preparation_provenance $score_provenance
    or return 1
    _checkpoint_complete $step
end

if set -q RESCORE_CHILD[1]
    set -q CHECKPOINT_STEP[1]
    or _fail "RESCORE_CHILD requires CHECKPOINT_STEP"
    string match -rq '^[1-9][0-9]*$' -- $CHECKPOINT_STEP
    or _fail "invalid child checkpoint step: $CHECKPOINT_STEP"
    _run_step $CHECKPOINT_STEP
    exit $status
end

set -l checkpoint_steps
if set -q CHECKPOINT_STEPS[1]
    set checkpoint_steps (string split ',' -- (string replace -a ' ' ',' -- $CHECKPOINT_STEPS))
else
    for source_root in (find $SOURCE_OUTPUT_BASE -mindepth 1 -maxdepth 1 -type d \
        -name 'dancegrpo_vbvr_pro_5b_checkpoint-*-cps-noise-0.7' | sort -V)
        set -a checkpoint_steps (string replace -r '^.*checkpoint-([0-9]+)-cps-noise-0\.7$' '$1' -- $source_root)
    end
end
test (count $checkpoint_steps) -gt 0; or _fail "no checkpoint source outputs found"
for step in $checkpoint_steps
    string match -rq '^[1-9][0-9]*$' -- $step
    or _fail "invalid checkpoint step: $step"
    test -d (_source_root_for_step $step)
    or _fail "source output is missing for checkpoint-$step"
end

mkdir -p $OUTPUT_BASE $EVAL_LOG_DIR; or exit 1
echo "[rescore] checkpoints: $checkpoint_steps"
echo "[rescore] source prepared outputs: $SOURCE_OUTPUT_BASE"
echo "[rescore] required source generation: "$EXPECTED_GENERATION_WIDTH"x"$EXPECTED_GENERATION_HEIGHT"x81 at 16 FPS"
echo "[rescore] EvalKit: $EVALKIT_REV ($EVALKIT_SOURCE_SHA256)"
echo "[rescore] runtime: $scorer_dependency_versions"
echo "[rescore] concurrency: $MAX_PARALLEL jobs x $SCORE_WORKERS workers x $SCORE_THREADS_PER_WORKER threads"
echo "[rescore] output base: $OUTPUT_BASE"

set -l wave_start 1
while test $wave_start -le (count $checkpoint_steps)
    set -l wave_steps
    for offset in (seq 0 (math $MAX_PARALLEL - 1))
        set -l position (math $wave_start + $offset)
        if test $position -le (count $checkpoint_steps)
            set -a wave_steps $checkpoint_steps[$position]
        end
    end

    set -l running_steps
    set -l running_pids
    set -l running_logs
    for step in $wave_steps
        set -l log_path $EVAL_LOG_DIR/checkpoint-$step-rescore-e140.log
        echo "[start] "(date --iso-8601=seconds)" checkpoint-$step log=$log_path"
        env WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED=1 RESCORE_CHILD=1 CHECKPOINT_STEP=$step \
            fish $RESCORE_SCRIPT >$log_path 2>&1 &
        set -a running_steps $step
        set -a running_pids $last_pid
        set -a running_logs $log_path
    end

    set -l wave_failed 0
    for index in (seq (count $running_pids))
        wait $running_pids[$index]
        set -l wait_status $status
        if set -q DRY_RUN[1]; and test $wait_status -eq 0
            echo "[dry-run] checkpoint-$running_steps[$index] source/provenance checks passed"
        else if test $wait_status -ne 0; or not _checkpoint_complete $running_steps[$index]
            set wave_failed 1
            echo "[error] checkpoint-$running_steps[$index] scorer-only evaluation failed; log=$running_logs[$index]" >&2
            tail -n 120 $running_logs[$index] >&2
        else
            echo "[done]  "(date --iso-8601=seconds)" checkpoint-$running_steps[$index] log=$running_logs[$index]"
        end
    end
    test $wave_failed -eq 0; or _fail "scorer-only checkpoint wave failed: $wave_steps"
    set wave_start (math $wave_start + $MAX_PARALLEL)
end

if not set -q DRY_RUN[1]
    $PYTHON -c '
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
prepared_name = sys.argv[3]
rows = []
for run in output_root.glob("dancegrpo_vbvr_pro_5b_checkpoint-*-cps-noise-0.7"):
    step = int(run.name.split("checkpoint-", 1)[1].split("-", 1)[0])
    source_run = source_root / run.name
    filename = f"{prepared_name}_vbvr_results.json"
    old_summary = json.loads((source_run / "scores" / filename).read_text())["summary"]
    new_summary = json.loads((run / "scores" / filename).read_text())["summary"]
    categories = new_summary["overall"]["by_category"]
    rows.append((
        step,
        new_summary["overall"]["mean_score"],
        new_summary["In_Domain"]["mean_score"],
        new_summary["Out_of_Domain"]["mean_score"],
        categories["Abstraction"],
        categories["Perception"],
        categories["Spatiality"],
        categories["Transformation"],
        categories["Knowledge"],
        old_summary["overall"]["mean_score"],
        old_summary["In_Domain"]["mean_score"],
        old_summary["Out_of_Domain"]["mean_score"],
    ))

score_header = (
    "checkpoint\toverall\tin_domain\tout_of_domain\tabstraction\tperception"
    "\tspatiality\ttransformation\tknowledge"
)
score_lines = [score_header]
migration_header = (
    "checkpoint\told_overall\tnew_overall\tdelta_overall"
    "\told_in_domain\tnew_in_domain\tdelta_in_domain"
    "\told_out_of_domain\tnew_out_of_domain\tdelta_out_of_domain"
)
migration_lines = [migration_header]
for row in sorted(rows):
    score_lines.append("\t".join((str(row[0]), *(f"{value:.6f}" for value in row[1:9]))))
    migration_lines.append("\t".join((
        str(row[0]),
        f"{row[9]:.6f}", f"{row[1]:.6f}", f"{row[1] - row[9]:+.6f}",
        f"{row[10]:.6f}", f"{row[2]:.6f}", f"{row[2] - row[10]:+.6f}",
        f"{row[11]:.6f}", f"{row[3]:.6f}", f"{row[3] - row[11]:+.6f}",
    )))
score_table = "\n".join(score_lines) + "\n"
migration_table = "\n".join(migration_lines) + "\n"
(output_root / "checkpoint_scores.tsv").write_text(score_table)
(output_root / "scorer_migration.tsv").write_text(migration_table)
print(score_table, end="")
print(migration_table, end="")
' $SOURCE_OUTPUT_BASE $OUTPUT_BASE $PREPARED_NAME
    or _fail "could not write scorer-only checkpoint summaries"
end

echo "[done] scorer-only e140 EvalKit sweep completed; no videos were generated or prepared"
