#!/usr/bin/env fish

# Evaluate one checkpoint from the Fujian manifest-RL run on the complete
# 500-sample VBVR-Pro bench. The default remains the training rollout policy
# (30-step Flow-CPS 0.7, CFG 1.0, seed 0), while callers may select another
# CPS coefficient or a matched 30-step Euler/UniPC ODE through environment
# variables. This keeps all non-sampler evaluation settings identical.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -q CHECKPOINT_STEP[1]
or _fail "set CHECKPOINT_STEP to a checkpoint number such as 100"
string match -rq '^[0-9]+$' -- $CHECKPOINT_STEP
or _fail "CHECKPOINT_STEP must be a positive integer: $CHECKPOINT_STEP"

set -q CHECKPOINT_ROOT[1]
or set -lx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_384x384x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian
set -lx CHECKPOINT $CHECKPOINT_ROOT/checkpoint-$CHECKPOINT_STEP
test -f $CHECKPOINT/high/.metadata
or _fail "checkpoint is missing high/.metadata: $CHECKPOINT"

set -q CONVERTED_MODEL[1]
or set -lx CONVERTED_MODEL storage/models/dcp_converted_5b/dancegrpo_vbvr_pro_5b_384x384x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_checkpoint-$CHECKPOINT_STEP

set -q GT_BASE[1]
or set -lx GT_BASE (realpath storage/datasets/vbvr-pro-eval-500)
set -q SPLIT_MANIFEST[1]
or set -lx SPLIT_MANIFEST $GT_BASE/split_manifest.json
test -d $GT_BASE; or _fail "GT_BASE does not exist: $GT_BASE"
test -f $SPLIT_MANIFEST; or _fail "split manifest does not exist: $SPLIT_MANIFEST"

set -l expected_sanitized_manifest_sha256 afab352e08c590c9f4b480ef314b37f6896eef6430f42ea6c0ce0494f2aa8c4e
set -l expected_source_manifest_sha256 326f7bda3743e9c66dc0c29445661a5dda4ad0cee4cb8838c3fcfd0c4a149deb
set -l expected_checksums_sha256 a67c534293724ddfc6657af755ab65e9b1354879deb2cfc47de22ede43942861
set -l actual_manifest_sha256 (sha256sum $SPLIT_MANIFEST | awk '{print $1}')
test "$actual_manifest_sha256" = "$expected_sanitized_manifest_sha256"
or _fail "sanitized split manifest mismatch: expected=$expected_sanitized_manifest_sha256 actual=$actual_manifest_sha256"
set -l actual_checksums_sha256 (sha256sum $GT_BASE/SHA256SUMS | awk '{print $1}')
test "$actual_checksums_sha256" = "$expected_checksums_sha256"
or _fail "dataset checksum manifest mismatch: expected=$expected_checksums_sha256 actual=$actual_checksums_sha256"

.venv/bin/python -c '
import json
import sys

config = json.load(open(sys.argv[1]))
expected = {
    "format": "vbvr_pro_flat_eval_v1",
    "repo_id": "pufanyi/vbvr-pro-eval-500",
    "samples": 500,
    "tasks": 100,
    "source_split_manifest_sha256": sys.argv[2],
    "sanitized_split_manifest_sha256": sys.argv[3],
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"dataset_config mismatch for {key}: expected={value!r} actual={config.get(key)!r}")
' $GT_BASE/dataset_config.json $expected_source_manifest_sha256 $expected_sanitized_manifest_sha256
or exit 1

if not set -q WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED[1]
    pushd $GT_BASE >/dev/null; or exit 1
    sha256sum -c SHA256SUMS --quiet
    set -l checksum_status $status
    popd >/dev/null; or exit 1
    test $checksum_status -eq 0; or _fail "VBVR-Pro eval snapshot failed SHA-256 verification"
end

set -q EVALKIT_REV[1]
or set -lx EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -q EVALKIT_SOURCE_SHA256[1]
or set -lx EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
set -q EVALKIT_DIR[1]
or set -lx EVALKIT_DIR storage/evalkits/vbvr-evalkit-interleave-main_v2-e140038f
set -q EASYOCR_ROOT[1]
or set -lx EASYOCR_ROOT storage/evalkits/easyocr-shared
set -q EASYOCR_SOURCE_MODELS[1]
or set -lx EASYOCR_SOURCE_MODELS $EASYOCR_ROOT/model

set -q HEIGHT[1]
or set -lx HEIGHT 384
set -q WIDTH[1]
or set -lx WIDTH 384
set -l native_shape "$HEIGHT"x"$WIDTH"x81

set -q GENERATION_MODE[1]
or set -lx GENERATION_MODE cps
set -q ODE_SOLVER[1]
or set -lx ODE_SOLVER unipc
set -q CPS_NOISE_LEVEL[1]
or set -lx CPS_NOISE_LEVEL 0.7
contains -- $GENERATION_MODE cps ode
or _fail "GENERATION_MODE must be cps or ode: $GENERATION_MODE"
contains -- $ODE_SOLVER euler unipc
or _fail "ODE_SOLVER must be euler or unipc: $ODE_SOLVER"

set -l sampler_label
if test "$GENERATION_MODE" = cps
    .venv/bin/python -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if 0 <= x <= 1 else 1)' $CPS_NOISE_LEVEL
    or _fail "CPS_NOISE_LEVEL must be in [0, 1]: $CPS_NOISE_LEVEL"
    set sampler_label cps-noise-$CPS_NOISE_LEVEL
else
    set sampler_label $ODE_SOLVER-ode-30steps-cfg1
end

set -q OUTPUT_BASE[1]
or set -l OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_$native_shape"_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028"
set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT $OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$CHECKPOINT_STEP-$sampler_label
set -q GENERATED_DIR[1]
or set -lx GENERATED_DIR $OUTPUT_ROOT/generated_$native_shape
set -q PREPARED_DIR[1]
or set -lx PREPARED_DIR $OUTPUT_ROOT/eval_1024x1024_81f_fps16_5p0625s
set -q SCORE_DIR[1]
or set -lx SCORE_DIR $OUTPUT_ROOT/scores

set -lx NUM_INFERENCE_STEPS 30
set -lx GUIDANCE_SCALE 1.0
set -lx SEED 0
set -lx NUM_FRAMES 81
set -lx INFER_FPS 16
set -lx MAX_DURATION 5.0625
set -lx EXPECTED_VIDEOS 500

set -l script_dir (dirname (status filename))
fish $script_dir/../vbvr_pro_5b_main_v2.fish $argv
set -l pipeline_status $status
if test $pipeline_status -ne 0
    exit $pipeline_status
end
if set -q DRY_RUN[1]; or set -q CONVERSION_ONLY[1]
    exit 0
end

set -l prepared_name (basename (string trim -r -c / -- $PREPARED_DIR))
set -l result_json $SCORE_DIR/$prepared_name"_vbvr_results.json"
set -q TASK_SCORE_XLSX[1]
or set TASK_SCORE_XLSX $SCORE_DIR/$prepared_name"_task_scores.xlsx"
set -q FINAL_SCORES_TXT[1]
or set FINAL_SCORES_TXT $OUTPUT_ROOT/final_scores.txt

.venv/bin/python -m src.cli.export_vbvr_task_scores $result_json \
    --output $TASK_SCORE_XLSX \
    --summary-output $FINAL_SCORES_TXT \
    --expected-samples 500 \
    --expected-tasks 100
or exit 1

echo "[done] task score workbook: $TASK_SCORE_XLSX"
echo "[done] concise scores: $FINAL_SCORES_TXT"
