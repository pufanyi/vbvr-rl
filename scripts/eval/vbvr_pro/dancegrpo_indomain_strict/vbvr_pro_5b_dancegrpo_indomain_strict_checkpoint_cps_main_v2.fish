#!/usr/bin/env fish

# Complete VBVR-Pro evaluation for one strict In-Domain DanceGRPO checkpoint
# with Flow-CPS. Set CHECKPOINT_STEP and CPS_NOISE_LEVEL, or use a fixed
# checkpoint/noise wrapper.

source (dirname (status filename))/../../../lib/env.fish

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

set -q CHECKPOINT_ROOT[1]
or set -lx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6_indomain_strict

set -l checkpoint_slug dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6_indomain_strict_checkpoint-$CHECKPOINT_STEP
set -q CONVERTED_MODEL[1]
or set -lx CONVERTED_MODEL storage/models/dcp_converted_5b/$checkpoint_slug

# Use a separate evaluation root while preserving the standard run names used
# by the VBVR-Pro summary tooling.
set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_indomain_strict/dancegrpo_vbvr_pro_5b_checkpoint-$CHECKPOINT_STEP-cps-noise-$CPS_NOISE_LEVEL

exec fish (dirname (status filename))/../dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_cps_main_v2.fish $argv
