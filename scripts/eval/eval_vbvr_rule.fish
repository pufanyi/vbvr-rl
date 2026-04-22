#!/usr/bin/env fish

# Score pre-generated videos against VBVR-Bench using the vendored
# rule-based EvalKit (third_party/VBVR-EvalKit/run_evaluation.py).
#
# Assumes videos already exist under each MODEL_OUT with layout:
#   $MODEL_OUT/{In-Domain_50,Out-of-Domain_50}/{task}/{00000..00004}.mp4
#
# Per-model results land in $MODEL_OUT/score/.
#
# Run from repo root:
#   fish scripts/eval/eval_vbvr_rule.fish

# ── Configuration ────────────────────────────────────────────────────
set MODEL_OUTPUTS \
    storage/eval_out/vbvr/sft_vbvr_checkpoint-12000
    # storage/eval_out/vbvr/correction_vbvr_checkpoint-3000 \

set GT_BASE      /mnt/umm/users/pufanyi/workspace/Wan-Trainer/data/vbvr/VBVR-Bench
set EVALKIT_DIR  third_party/VBVR-EvalKit
set DEVICE       cuda

# If your videos live under $MODEL_OUT/$SOURCE_SPLIT/{task}/ (e.g. Open_60/)
# instead of the In-Domain_50/ & Out-of-Domain_50/ layout EvalKit expects,
# set SOURCE_SPLIT and we'll symlink tasks into the correct split first.
# Leave empty to skip restructuring.
set SOURCE_SPLIT Open_60
# ─────────────────────────────────────────────────────────────────────

for MODEL_OUT in $MODEL_OUTPUTS
    if not test -d $MODEL_OUT
        echo "[skip] $MODEL_OUT does not exist"
        continue
    end

    set -l ABS_MODEL_OUT (realpath $MODEL_OUT)
    set -l SCORE_DIR $ABS_MODEL_OUT/score

    echo ""
    echo "==============================================================="
    echo "Scoring:  $MODEL_OUT"
    echo "Output:   $SCORE_DIR"
    echo "==============================================================="

    if test -n "$SOURCE_SPLIT"; and test -d $ABS_MODEL_OUT/$SOURCE_SPLIT
        uv run python scripts/eval/vbvr_restructure_to_evalkit.py \
            --model_out    $ABS_MODEL_OUT \
            --source_split $SOURCE_SPLIT
        or begin
            echo "[warn] restructure failed for $MODEL_OUT"
            continue
        end
    end

    uv run python $EVALKIT_DIR/run_evaluation.py \
        --model_path $ABS_MODEL_OUT \
        --gt_base    $GT_BASE \
        --output_dir $SCORE_DIR \
        --device     $DEVICE
    or echo "[warn] rule scoring failed for $MODEL_OUT"
end

echo ""
echo "Done."
