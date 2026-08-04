#!/usr/bin/env fish

# Single-variable spatial-resolution comparison for the preconverted DiffSynth
# TI2V-5B step-35500 model. This keeps the 81-frame/16-FPS/50-step UniPC ODE
# protocol fixed and changes only native generation from 256x256 to the
# checkpoint's 512x512 training resolution. Every generated frame is then
# resized without crop to the 1024x1024 scorer canvas.

source (dirname (status filename))/../../lib/env.fish

set -lx PRECONVERTED_MODEL 1
set -q CHECKPOINT[1]
or set -lx CHECKPOINT /mnt/umm/users/wangruisi/01-project/mllm/DiffSynth-Studio/wan2.2-TI2V-5B_260715_vbvr_pro/step-35500.safetensors
set -q BASE_MODEL[1]
or set -lx BASE_MODEL storage/models/Wan2.2-TI2V-5B-Diffusers
set -q CONVERTED_MODEL[1]
or set -lx CONVERTED_MODEL storage/models/diffsynth_converted_5b/wan2.2-TI2V-5B_260715_vbvr_pro_step-35500
set -q CONVERSION_PROVENANCE[1]
or set -lx CONVERSION_PROVENANCE $CONVERTED_MODEL/conversion_metadata.json

set -lx GENERATION_MODE ode
set -lx ODE_SOLVER unipc
set -lx NUM_INFERENCE_STEPS 50
set -lx GUIDANCE_SCALE 5.0
set -lx SEED 0
set -lx HEIGHT 512
set -lx WIDTH 512
set -lx NUM_FRAMES 81
set -lx INFER_FPS 16

set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_evalkit_4cc7d028/diffsynth_step35500-unipc-50steps-cfg5-512x512-81f-fps16
set -q GENERATED_DIR[1]
or set -lx GENERATED_DIR $OUTPUT_ROOT/generated_512x512x81
set -q PREPARED_DIR[1]
or set -lx PREPARED_DIR $OUTPUT_ROOT/eval_1024x1024_81f_fps16_5p0625s
set -q SCORE_DIR[1]
or set -lx SCORE_DIR $OUTPUT_ROOT/scores
set -lx MAX_DURATION 5.0625

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
