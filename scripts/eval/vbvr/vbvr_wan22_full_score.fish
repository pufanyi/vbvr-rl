#!/usr/bin/env fish

# Score pre-generated Wan2.2 full-model videos under
# storage/eval_out/vbvr_wan22_full/videos against VBVR-Bench using the
# vendored rule-based EvalKit (third_party/VBVR-EvalKit/run_evaluation.py).
#
# Videos already follow the EvalKit layout:
#   $MODEL_OUT/{In-Domain_50,Out-of-Domain_50}/{task}/{00000..}.mp4
#
# Per-model results land in $MODEL_OUT/score/.
#
# Run from repo root:
#   fish scripts/eval/vbvr/vbvr_wan22_full_score.fish

source (dirname (status filename))/../../lib/env.fish

# ── Configuration ────────────────────────────────────────────────────
set MODEL_OUTPUTS \
    storage/eval_out/vbvr_wan22_full/videos

set GT_BASE      storage/datasets/VBVR-Bench
set EVALKIT_DIR  third_party/VBVR-EvalKit
set DEVICE       cuda
set NUM_WORKERS  32

# Layout is already EvalKit-native; leave SOURCE_SPLIT empty to skip restructure.
set SOURCE_SPLIT
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
        python -m src.eval.vbvr_restructure_to_evalkit \
            --model_out    $ABS_MODEL_OUT \
            --source_split $SOURCE_SPLIT
        or begin
            echo "[warn] restructure failed for $MODEL_OUT"
            continue
        end
    end

    python -m src.eval.vbvr_run_evaluation_parallel \
        --model_path  $ABS_MODEL_OUT \
        --gt_base     $GT_BASE \
        --output_dir  $SCORE_DIR \
        --device      $DEVICE \
        --num_workers $NUM_WORKERS
    or echo "[warn] rule scoring failed for $MODEL_OUT"
end

echo ""
echo "Done."
