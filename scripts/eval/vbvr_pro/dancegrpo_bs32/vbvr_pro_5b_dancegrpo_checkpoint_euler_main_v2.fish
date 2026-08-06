#!/usr/bin/env fish

# Complete VBVR-Pro evaluation for one DanceGRPO checkpoint with deterministic
# first-order rectified-flow Euler sampling.

source (dirname (status filename))/../../../lib/env.fish

set -q CHECKPOINT_STEP[1]
or begin
    echo "[error] set CHECKPOINT_STEP to a checkpoint number such as 300" >&2
    exit 1
end

set -lx GENERATION_MODE ode
set -lx ODE_SOLVER euler
set -lx NUM_INFERENCE_STEPS 50
set -lx GUIDANCE_SCALE 5.0
set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_evalkit_4cc7d028/dancegrpo_vbvr_pro_5b_checkpoint-$CHECKPOINT_STEP-euler

exec fish (dirname (status filename))/vbvr_pro_5b_dancegrpo_checkpoint_main_v2.fish $argv
