#!/usr/bin/env fish

# VBVR-Bench evaluation pipeline.
#
# For each checkpoint:
#   1. Build eval JSON from the VBVR-Bench GT directory (once, cached).
#   2. Generate videos with src.cli.eval_i2v (multi-GPU).
#   3. Score generated videos immediately. Default is rule-based VBVR-EvalKit.
#
#   fish scripts/eval/eval_vbvr.fish

source (dirname (status filename))/../activate_venv.fish

# ── Configuration ────────────────────────────────────────────────────
set CHECKPOINTS \
    storage/checkpoints/sft_vbvr_fixed/checkpoint-4000
    # storage/checkpoints/sft_vbvr_5e-6/checkpoint-10000
    # storage/checkpoints/sft_vbvr/checkpoint-12000
    # storage/models/Wan2.2-I2V-A14B-Diffusers
    # storage/checkpoints/sft_vbvr/checkpoint-4000
    # storage/checkpoints/sft_vbvr/checkpoint-8000
    # storage/checkpoints/sft_vbvr/checkpoint-12000
    # storage/checkpoints/correction_vbvr/checkpoint-3000 \
    # storage/checkpoints/sft_maze/checkpoint-2000
    # storage/checkpoints/cos_maze/checkpoint-4000

set NUM_GPUS 8
set GT_BASE /mnt/umm/users/pufanyi/workspace/Wan-Trainer/data/vbvr/VBVR-Bench
set EVALKIT_DIR third_party/VBVR-EvalKit
set EVAL_JSON storage/eval_out/vbvr/vbvr_eval.json
set SPLIT Open_60                          # generation directory label; rule scoring is split-agnostic after restructure
set TASKS                                   # leave empty to evaluate all tasks
set EXTRA_INFER_ARGS --num_inference_steps 50
set SKIP_GENERATION                         # set to any value to only re-score existing videos
set SKIP_SCORING                            # set to any value to only generate videos

# Scoring backend: "vlm" (our src.cli.eval_vbvr) or "rule" (VBVR-EvalKit rule-based)
set JUDGE rule
set VLM_MODEL google/gemma-4-26B-A4B-it
set VLM_NUM_FRAMES 6
set VLM_OUTPUT_DIR storage/eval_out/vbvr_vlm
set RULE_DEVICE cuda
set RULE_NUM_WORKERS 64
set RULE_SOURCE_SPLIT $SPLIT                # set empty if videos already use In-Domain_50 / Out-of-Domain_50 layout
# ─────────────────────────────────────────────────────────────────────

# ── 1. Build eval JSON (cached) ──────────────────────────────────────
if not test -f $EVAL_JSON
    set -l build_args --gt_base $GT_BASE --output $EVAL_JSON --split $SPLIT
    if test (count $TASKS) -gt 0
        set -a build_args --tasks $TASKS
    end
    python -m src.eval.build_vbvr_eval_json $build_args
    or exit 1
else
    echo "Reusing eval JSON at $EVAL_JSON"
end

# ── 2. Generate + 3. Score for each checkpoint ───────────────────────
for CKPT in $CHECKPOINTS
    set -l CKPT_NAME (string replace -a / _ (string trim -r -c / $CKPT | string replace -r '.*storage/checkpoints/' ''))
    set -l MODEL_OUT_DIR storage/eval_out/vbvr/$CKPT_NAME

    # A DCP training checkpoint has .metadata at root (flat layout) or under
    # high/ and/or low/ (expert-parallel layout). Anything else is assumed to
    # be a plain diffusers directory we should load as --model_path directly.
    set -l IS_DCP 0
    if test -f $CKPT/.metadata; or test -f $CKPT/high/.metadata; or test -f $CKPT/low/.metadata
        set IS_DCP 1
    end

    set -l infer_args --eval_json $EVAL_JSON --output_dir $MODEL_OUT_DIR
    if test $IS_DCP -eq 1
        set -a infer_args --checkpoint $CKPT --use_ema
    else
        set -a infer_args --model_path $CKPT
    end
    if test (count $EXTRA_INFER_ARGS) -gt 0
        set -a infer_args $EXTRA_INFER_ARGS
    end

    echo ""
    echo "==============================================================="
    echo "Checkpoint:  $CKPT"
    if test $IS_DCP -eq 1
        echo "Mode:        DCP checkpoint (use_ema=on)"
    else
        echo "Mode:        plain diffusers model"
    end
    echo "Output dir:  $MODEL_OUT_DIR"
    echo "==============================================================="

    if not set -q SKIP_GENERATION[1]
        if test $NUM_GPUS -gt 1
            torchrun --nproc_per_node=$NUM_GPUS -m src.cli.eval_i2v $infer_args
            or continue
        else
            python -m src.cli.eval_i2v $infer_args
            or continue
        end
    end

    if not set -q SKIP_SCORING[1]
        set -l ABS_MODEL_OUT (realpath $MODEL_OUT_DIR)

        switch $JUDGE
            case vlm
                set -l score_args \
                    --model_output $ABS_MODEL_OUT \
                    --gt_base $GT_BASE \
                    --output_dir $VLM_OUTPUT_DIR \
                    --judge_model $VLM_MODEL \
                    --num_frames $VLM_NUM_FRAMES
                if test (count $TASKS) -gt 0
                    set -a score_args --tasks $TASKS
                end
                if test $NUM_GPUS -gt 1
                    torchrun --nproc_per_node=$NUM_GPUS -m src.cli.eval_vbvr $score_args
                    or echo "[warn] VLM scoring failed for $CKPT"
                else
                    python -m src.cli.eval_vbvr $score_args
                    or echo "[warn] VLM scoring failed for $CKPT"
                end

            case rule
                set -l SCORE_DIR $ABS_MODEL_OUT/score
                if test (count $TASKS) -gt 0
                    echo "[warn] rule scorer has no --tasks flag; scoring all videos found under $ABS_MODEL_OUT"
                end

                if test -n "$RULE_SOURCE_SPLIT"; and test -d $ABS_MODEL_OUT/$RULE_SOURCE_SPLIT
                    python -m src.eval.vbvr_restructure_to_evalkit \
                        --model_out    $ABS_MODEL_OUT \
                        --source_split $RULE_SOURCE_SPLIT
                    or begin
                        echo "[warn] restructure failed for $CKPT"
                        continue
                    end
                end

                python -m src.eval.vbvr_run_evaluation_parallel \
                    --model_path  $ABS_MODEL_OUT \
                    --gt_base     $GT_BASE \
                    --output_dir  $SCORE_DIR \
                    --device      $RULE_DEVICE \
                    --num_workers $RULE_NUM_WORKERS
                or echo "[warn] rule scoring failed for $CKPT"

            case '*'
                echo "[error] unknown JUDGE=$JUDGE (expected 'vlm' or 'rule')"
        end
    end
end

echo ""
echo "Done."
