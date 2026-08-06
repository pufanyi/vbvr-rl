#!/usr/bin/env fish

# Complete VBVR-Pro main_v2 evaluation and reporting for one DanceGRPO
# checkpoint. Set CHECKPOINT_STEP or use one of the fixed-step wrappers.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -q CHECKPOINT_STEP[1]
or _fail "set CHECKPOINT_STEP to a checkpoint number such as 300"
string match -rq '^[0-9]+$' -- $CHECKPOINT_STEP
or _fail "CHECKPOINT_STEP must be a positive integer: $CHECKPOINT_STEP"

set -q CHECKPOINT_ROOT[1]
or set CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32
set -q CHECKPOINT[1]
or set -lx CHECKPOINT $CHECKPOINT_ROOT/checkpoint-$CHECKPOINT_STEP
test -d $CHECKPOINT
or _fail "checkpoint directory does not exist: $CHECKPOINT"

set -l checkpoint_slug dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_checkpoint-$CHECKPOINT_STEP
set -q CONVERTED_MODEL[1]
or set -lx CONVERTED_MODEL storage/models/dcp_converted_5b/$checkpoint_slug
set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_evalkit_4cc7d028/dancegrpo_vbvr_pro_5b_checkpoint-$CHECKPOINT_STEP
set -q PREPARED_DIR[1]
or set -lx PREPARED_DIR $OUTPUT_ROOT/eval_1024x1024_161f_5s
set -q SCORE_DIR[1]
or set -lx SCORE_DIR $OUTPUT_ROOT/scores

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
