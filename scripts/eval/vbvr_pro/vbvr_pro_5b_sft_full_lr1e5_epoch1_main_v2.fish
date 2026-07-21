#!/usr/bin/env fish

# Evaluate the SFT 5B full-FT epoch-1 checkpoint on the 500-sample VBVR-Pro
# main_v2 subset. The shared launcher owns conversion, 8-GPU generation,
# 1024x1024x161 preparation, scoring, validation, resume, provenance, and the
# final per-task Excel report.

source (dirname (status filename))/../../lib/env.fish

set -q CHECKPOINT[1]
or set -lx CHECKPOINT storage/checkpoints/sft_vbvr_5b_256x256x161_full_lr_1e-5/checkpoint-epoch1

# Keep this separate from the older unprovenanced conversion of the same
# checkpoint. A fresh conversion makes the exact source and options auditable.
set -q CONVERTED_MODEL[1]
or set -lx CONVERTED_MODEL storage/models/dcp_converted_5b/sft_vbvr_5b_256x256x161_full_lr_1e-5_checkpoint-epoch1-main-v2

set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_evalkit_eb977da6/sft_vbvr_5b_256x256x161_full_lr_1e-5_checkpoint-epoch1

set -q PREPARED_DIR[1]
or set -lx PREPARED_DIR $OUTPUT_ROOT/eval_1024x1024_161f_5s

set -q SCORE_DIR[1]
or set -lx SCORE_DIR $OUTPUT_ROOT/scores

set -l script_dir (dirname (status filename))
fish $script_dir/vbvr_pro_5b_main_v2.fish $argv
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
