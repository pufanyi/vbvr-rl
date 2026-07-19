#!/usr/bin/env fish

set -lx CHECKPOINT_STEP 300
set -lx CPS_NOISE_LEVEL 0.7
set -lx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6
set -lx CONVERTED_MODEL storage/models/dcp_converted_5b/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6_checkpoint-300
set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2/dancegrpo_vbvr_pro_5b_bs32_lr_1e-6_checkpoint-300-cps-noise-0.7
exec fish (dirname (status filename))/../dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_cps_main_v2.fish $argv
