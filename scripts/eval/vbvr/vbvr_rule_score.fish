#!/usr/bin/env fish

# Score videos against VBVR-Bench using the vendored rule-based EvalKit.
#
# Each entry in MODEL_OUTPUTS can be either:
#   1. A video output dir containing .mp4 files.
#   2. A DCP checkpoint / diffusers model dir. If no videos are found, this
#      script generates videos first, then scores them.
#
# Generation writes:
#   $MODEL_OUT/$SPLIT/{task}/{00000..00004}.mp4
# Scoring uses/restructures to:
#   $MODEL_OUT/{In-Domain_50,Out-of-Domain_50}/{task}/{00000..00004}.mp4
#
# Per-model results land in $MODEL_OUT/score/.
#
# Run from repo root:
#   fish scripts/eval/vbvr/vbvr_rule_score.fish
#
# Runtime overrides, useful on memory-constrained eval nodes:
#   NUM_GPUS=4 INFER_NUM_FRAMES=49 fish scripts/eval/vbvr/vbvr_rule_score.fish

source (dirname (status filename))/../../lib/env.fish

set -e RANK
set -e WORLD_SIZE
set -e MASTER_ADDR
set -e MASTER_PORT
set -e LOCAL_RANK
set -e LOCAL_WORLD_SIZE

# ── Configuration ────────────────────────────────────────────────────
if not set -q MODEL_OUTPUTS[1]
    set MODEL_OUTPUTS \
        storage/checkpoints/sft_vbvr_fixed/checkpoint-2000
        # storage/eval_out/vbvr/sft_vbvr_5e-6_checkpoint-10000
        # storage/eval_out/vbvr/sft_vbvr_checkpoint-8000
        # storage/eval_out/vbvr/correction_vbvr_checkpoint-3000 \
        # storage/eval_out/vbvr/sft_vbvr_checkpoint-4000 \
        # storage/eval_out/vbvr/sft_vbvr_checkpoint-12000
end

set -q NUM_GPUS[1]; or set NUM_GPUS 8
set -q GT_BASE[1]; or set GT_BASE /mnt/umm/users/pufanyi/workspace/Wan-Trainer/data/vbvr/VBVR-Bench
set -q EVALKIT_DIR[1]; or set EVALKIT_DIR third_party/VBVR-EvalKit
set -q EVAL_JSON[1]; or set EVAL_JSON storage/eval_out/vbvr/vbvr_eval.json
set -q SPLIT[1]; or set SPLIT Open_60
set -q TASKS[1]; or set TASKS
set -q EXTRA_INFER_ARGS[1]; or set EXTRA_INFER_ARGS # e.g. set -gx EXTRA_INFER_ARGS --guidance_scale 4.0
set -q SKIP_GENERATION[1]; or set SKIP_GENERATION # set to any value to only score existing videos

# Optional eval_i2v overrides. Steps default to 50 for eval runs.
set -q INFER_MAX_AREA[1]; or set INFER_MAX_AREA
set -q INFER_NUM_FRAMES[1]; or set INFER_NUM_FRAMES
set -q INFER_NUM_INFERENCE_STEPS[1]; or set INFER_NUM_INFERENCE_STEPS 50
set -q INFER_GUIDANCE_SCALE[1]; or set INFER_GUIDANCE_SCALE
set -q INFER_FPS[1]; or set INFER_FPS
set -q INFER_SEED[1]; or set INFER_SEED
set INFER_DIST_BACKEND nccl
set -q INFER_DIST_TIMEOUT_MINUTES[1]; or set INFER_DIST_TIMEOUT_MINUTES

set -q DEVICE[1]; or set DEVICE cuda
set -q NUM_WORKERS[1]; or set NUM_WORKERS 64

# If your videos live under $MODEL_OUT/$SOURCE_SPLIT/{task}/ (e.g. Open_60/)
# instead of the In-Domain_50/ & Out-of-Domain_50/ layout EvalKit expects,
# set SOURCE_SPLIT and we'll symlink tasks into the correct split first.
# Leave empty to skip restructuring.
set -q SOURCE_SPLIT[1]; or set SOURCE_SPLIT $SPLIT
# ─────────────────────────────────────────────────────────────────────

# ── 1. Build eval JSON if generation is needed ───────────────────────
if not test -f $EVAL_JSON
    set -l build_args --gt_base $GT_BASE --output $EVAL_JSON --layout split --split $SPLIT
    if test (count $TASKS) -gt 0
        set -a build_args --tasks $TASKS
    end
    python -m src.eval.build_vbvr_eval_json $build_args
    or exit 1
else
    echo "Reusing eval JSON at $EVAL_JSON"
end

# ── 2. Generate if needed + 3. Score ─────────────────────────────────
for TARGET in $MODEL_OUTPUTS
    if not test -d $TARGET
        echo "[skip] $TARGET does not exist"
        continue
    end

    set -l ABS_TARGET (realpath $TARGET)
    set -l TARGET_VIDEO (find $ABS_TARGET -type f -name '*.mp4' -print -quit)
    set -l MODEL_OUT $TARGET
    set -l NEED_GENERATION 0

    if test -z "$TARGET_VIDEO"
        set -l TARGET_NAME (string replace -a / _ (string trim -r -c / $TARGET | string replace -r '.*storage/checkpoints/' ''))
        set MODEL_OUT storage/eval_out/vbvr/$TARGET_NAME

        set -l GENERATED_VIDEO
        if test -d $MODEL_OUT
            set GENERATED_VIDEO (find $MODEL_OUT -type f -name '*.mp4' -print -quit)
        end

        if test -z "$GENERATED_VIDEO"
            set NEED_GENERATION 1
        else
            echo "[info] no videos under $TARGET; using existing generated output $MODEL_OUT"
        end
    end

    if test $NEED_GENERATION -eq 1
        if set -q SKIP_GENERATION[1]
            echo "[skip] no videos found for $TARGET and SKIP_GENERATION is set"
            continue
        end

        set -l IS_DCP 0
        if test -f $TARGET/.metadata; or test -f $TARGET/high/.metadata; or test -f $TARGET/low/.metadata
            set IS_DCP 1
        end

        if test $IS_DCP -eq 0; and not test -f $TARGET/model_index.json
            echo "[skip] no videos found for $TARGET, and it is not a DCP checkpoint or diffusers model dir"
            continue
        end

        set -l infer_args --eval_json $EVAL_JSON --output_dir $MODEL_OUT
        if test $IS_DCP -eq 1
            set -a infer_args --checkpoint $TARGET --use_ema
        else
            set -a infer_args --model_path $TARGET
        end
        if test -n "$INFER_MAX_AREA"
            set -a infer_args --max_area $INFER_MAX_AREA
        end
        if test -n "$INFER_NUM_FRAMES"
            set -a infer_args --num_frames $INFER_NUM_FRAMES
        end
        if test -n "$INFER_NUM_INFERENCE_STEPS"
            set -a infer_args --num_inference_steps $INFER_NUM_INFERENCE_STEPS
        end
        if test -n "$INFER_GUIDANCE_SCALE"
            set -a infer_args --guidance_scale $INFER_GUIDANCE_SCALE
        end
        if test -n "$INFER_FPS"
            set -a infer_args --fps $INFER_FPS
        end
        if test -n "$INFER_SEED"
            set -a infer_args --seed $INFER_SEED
        end
        if test -n "$INFER_DIST_BACKEND"
            set -a infer_args --dist_backend $INFER_DIST_BACKEND
        end
        if test -n "$INFER_DIST_TIMEOUT_MINUTES"
            set -a infer_args --dist_timeout_minutes $INFER_DIST_TIMEOUT_MINUTES
        end
        if test (count $EXTRA_INFER_ARGS) -gt 0
            set -a infer_args $EXTRA_INFER_ARGS
        end

        echo ""
        echo "==============================================================="
        echo "Generating: $TARGET"
        echo "Output:     $MODEL_OUT"
        echo "GPUs:       $NUM_GPUS"
        echo "==============================================================="

        if test $NUM_GPUS -gt 1
            torchrun --nproc_per_node=$NUM_GPUS -m src.cli.eval_i2v $infer_args
            or begin
                echo "[warn] generation failed for $TARGET"
                continue
            end
        else
            python -m src.cli.eval_i2v $infer_args
            or begin
                echo "[warn] generation failed for $TARGET"
                continue
            end
        end
    end

    if not test -d $MODEL_OUT
        echo "[skip] $MODEL_OUT does not exist"
        continue
    end

    set -l ABS_MODEL_OUT (realpath $MODEL_OUT)
    set -l OUTPUT_VIDEO (find $ABS_MODEL_OUT -type f -name '*.mp4' -print -quit)
    if test -z "$OUTPUT_VIDEO"
        echo "[skip] no videos found under $ABS_MODEL_OUT"
        continue
    end

    set -l SCORE_DIR $ABS_MODEL_OUT/score

    echo ""
    echo "==============================================================="
    echo "Scoring:  $ABS_MODEL_OUT"
    echo "Output:   $SCORE_DIR"
    echo "==============================================================="

    if test -n "$SOURCE_SPLIT"; and test -d $ABS_MODEL_OUT/$SOURCE_SPLIT
        python -m src.eval.vbvr_restructure_to_evalkit \
            --model_out $ABS_MODEL_OUT \
            --source_split $SOURCE_SPLIT
        or begin
            echo "[warn] restructure failed for $MODEL_OUT"
            continue
        end
    end

    python -m src.eval.vbvr_run_evaluation_parallel \
        --model_path $ABS_MODEL_OUT \
        --gt_base $GT_BASE \
        --output_dir $SCORE_DIR \
        --device $DEVICE \
        --num_workers $NUM_WORKERS
    or echo "[warn] rule scoring failed for $MODEL_OUT"
end

echo ""
echo "Done."
