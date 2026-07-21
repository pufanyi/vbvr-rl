#!/usr/bin/env fish

set -lx CHECKPOINT_STEP 200
set -lx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6
set -lx CONVERTED_MODEL storage/models/dcp_converted_5b/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6_checkpoint-200
set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_evalkit_eb977da6/dancegrpo_vbvr_pro_5b_bs32_lr_1e-6_checkpoint-200
exec fish (dirname (status filename))/../dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_main_v2.fish $argv
