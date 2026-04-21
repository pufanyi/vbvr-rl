#!/usr/bin/env fish

# VBVR-Bench evaluation pipeline.
#
# For each checkpoint:
#   1. Build eval JSON from the VBVR-Bench GT directory (once, cached).
#   2. Generate videos with src.cli.eval_i2v (multi-GPU).
#   3. Score with VBVR-EvalKit's run_evaluation_video_icml.py.
#
#   fish scripts/eval/eval_vbvr.fish

# ── Configuration ────────────────────────────────────────────────────
set CHECKPOINTS \
    storage/checkpoints/correction_vbvr/checkpoint-3000
    # storage/checkpoints/sft_maze/checkpoint-2000
    # storage/checkpoints/cos_maze/checkpoint-4000

set NUM_GPUS 8
set GT_BASE /mnt/umm/users/pufanyi/workspace/Wan-Trainer/data/vbvr/VBVR-Bench
set EVALKIT_DIR /mnt/aigc/xujunxiang/Code/VBVR-Bench/VBVR-EvalKit
set EVAL_JSON storage/eval_out/vbvr/vbvr_eval.json
set SPLIT Open_60                          # directory label only; scoring is split-agnostic
set TASKS                                   # leave empty to evaluate all tasks
set EXTRA_INFER_ARGS                        # e.g. --num_inference_steps 30
set SKIP_GENERATION                         # set to any value to only re-score existing videos
set SKIP_SCORING                            # set to any value to only generate videos

# Scoring backend: "vlm" (our src.cli.eval_vbvr) or "rule" (VBVR-EvalKit rule-based)
set JUDGE vlm
set VLM_MODEL google/gemma-4-26B-A4B-it
set VLM_NUM_FRAMES 6
set VLM_OUTPUT_DIR storage/eval_out/vbvr_vlm
# ─────────────────────────────────────────────────────────────────────

# ── 1. Build eval JSON (cached) ──────────────────────────────────────
if not test -f $EVAL_JSON
    set -l build_args --gt_base $GT_BASE --output $EVAL_JSON --split $SPLIT
    if test (count $TASKS) -gt 0
        set -a build_args --tasks $TASKS
    end
    uv run python scripts/eval/build_vbvr_eval_json.py $build_args
    or exit 1
else
    echo "Reusing eval JSON at $EVAL_JSON"
end

# ── 2. Generate + 3. Score for each checkpoint ───────────────────────
for CKPT in $CHECKPOINTS
    set -l CKPT_NAME (string replace -a / _ (string trim -r -c / $CKPT | string replace -r '.*storage/checkpoints/' ''))
    set -l MODEL_OUT_DIR storage/eval_out/vbvr/$CKPT_NAME

    echo ""
    echo "==============================================================="
    echo "Checkpoint:  $CKPT"
    echo "Output dir:  $MODEL_OUT_DIR"
    echo "==============================================================="

    if not set -q SKIP_GENERATION[1]
        if test $NUM_GPUS -gt 1
            uv run torchrun --nproc_per_node=$NUM_GPUS -m src.cli.eval_i2v \
                --eval_json $EVAL_JSON \
                --output_dir $MODEL_OUT_DIR \
                --checkpoint $CKPT \
                --use_ema $EXTRA_INFER_ARGS
            or continue
        else
            uv run python -m src.cli.eval_i2v \
                --eval_json $EVAL_JSON \
                --output_dir $MODEL_OUT_DIR \
                --checkpoint $CKPT \
                --use_ema $EXTRA_INFER_ARGS
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
                uv run python -m src.cli.eval_vbvr $score_args
                or echo "[warn] VLM scoring failed for $CKPT"

            case rule
                set -l SCORE_DIR $ABS_MODEL_OUT/score
                set -l score_args --model_path $ABS_MODEL_OUT --gt_base $GT_BASE --output_dir $SCORE_DIR
                if test (count $TASKS) -gt 0
                    set -a score_args --tasks $TASKS
                end
                pushd $EVALKIT_DIR
                uv run python run_evaluation_video_icml.py $score_args
                set -l rc $status
                popd
                test $rc -eq 0; or echo "[warn] rule scoring failed for $CKPT (rc=$rc)"

            case '*'
                echo "[error] unknown JUDGE=$JUDGE (expected 'vlm' or 'rule')"
        end
    end
end

echo ""
echo "Done."
