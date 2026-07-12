#!/usr/bin/env fish

# Complete VBVR-Pro evaluation for one DanceGRPO checkpoint with Flow-CPS.
# Set CHECKPOINT_STEP and CPS_NOISE_LEVEL, or use a fixed checkpoint wrapper.

source (dirname (status filename))/../lib/env.fish

set -q CHECKPOINT_STEP[1]
or begin
    echo "[error] set CHECKPOINT_STEP to a checkpoint number such as 300" >&2
    exit 1
end
set -q CPS_NOISE_LEVEL[1]
or begin
    echo "[error] set CPS_NOISE_LEVEL to a value in [0, 1]" >&2
    exit 1
end

set -lx GENERATION_MODE cps
set -lx NUM_INFERENCE_STEPS 30
set -lx GUIDANCE_SCALE 1.0
set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2/dancegrpo_vbvr_pro_5b_checkpoint-$CHECKPOINT_STEP-cps-noise-$CPS_NOISE_LEVEL

exec fish (dirname (status filename))/vbvr_pro_5b_dancegrpo_checkpoint_main_v2.fish $argv
