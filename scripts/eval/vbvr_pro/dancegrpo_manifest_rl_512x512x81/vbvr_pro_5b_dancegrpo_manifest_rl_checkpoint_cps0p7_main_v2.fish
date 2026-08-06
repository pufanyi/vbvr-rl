#!/usr/bin/env fish

# Evaluate one checkpoint from the 512x512x81 manifest-RL run on the complete
# 500-sample VBVR-Pro bench. Generation matches the training rollout policy:
# 30-step Flow-CPS with noise level 0.7, CFG 1.0, and seed 0. Prepared videos
# retain all 81 frames at exactly 16 FPS on the 1024x1024 scorer canvas.

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
or set -lx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_1e-6_manifest_rl_evalkit_6fedd9d9_reward1024_fps16_8_nodes
set -lx CHECKPOINT $CHECKPOINT_ROOT/checkpoint-$CHECKPOINT_STEP
test -f $CHECKPOINT/high/.metadata
or _fail "checkpoint is missing high/.metadata: $CHECKPOINT"

set -q CONVERTED_MODEL[1]
or set -lx CONVERTED_MODEL storage/models/dcp_converted_5b/dancegrpo_vbvr_pro_5b_512x512x81_manifest_rl_checkpoint-$CHECKPOINT_STEP

set -q SPLIT_MANIFEST[1]
or set -lx SPLIT_MANIFEST /mnt/aigc/xujunxiang/Code/VBVR-Pro/scripts/split_manifest.json
set -l expected_manifest_sha256 326f7bda3743e9c66dc0c29445661a5dda4ad0cee4cb8838c3fcfd0c4a149deb
test -f $SPLIT_MANIFEST
or _fail "split manifest does not exist: $SPLIT_MANIFEST"
set -l actual_manifest_sha256 (sha256sum $SPLIT_MANIFEST | awk '{print $1}')
test "$actual_manifest_sha256" = "$expected_manifest_sha256"
or _fail "split manifest fingerprint mismatch: expected=$expected_manifest_sha256 actual=$actual_manifest_sha256"

set -q EVALKIT_REV[1]
or set -lx EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -q EVALKIT_SOURCE_SHA256[1]
or set -lx EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8

set -q OUTPUT_BASE[1]
or set -l OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_manifest_326f7bda_evalkit_4cc7d028
set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT $OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$CHECKPOINT_STEP-cps-noise-0.7
set -q GENERATED_DIR[1]
or set -lx GENERATED_DIR $OUTPUT_ROOT/generated_512x512x81
set -q PREPARED_DIR[1]
or set -lx PREPARED_DIR $OUTPUT_ROOT/eval_1024x1024_81f_fps16_5p0625s
set -q SCORE_DIR[1]
or set -lx SCORE_DIR $OUTPUT_ROOT/scores

set -lx GENERATION_MODE cps
set -lx CPS_NOISE_LEVEL 0.7
set -lx NUM_INFERENCE_STEPS 30
set -lx GUIDANCE_SCALE 1.0
set -lx SEED 0
set -lx HEIGHT 512
set -lx WIDTH 512
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
