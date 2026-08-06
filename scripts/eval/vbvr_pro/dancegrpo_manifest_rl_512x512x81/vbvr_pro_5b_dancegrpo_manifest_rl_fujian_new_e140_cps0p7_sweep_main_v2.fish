#!/usr/bin/env fish

# Evaluate every complete checkpoint from the Fujian native-512 manifest-RL
# run trained with the e140038f reward. Generation matches the rollout policy:
# 30-step Flow-CPS 0.7, CFG 1.0, seed 0, 512x512x81, and exact 16 FPS.

source (dirname (status filename))/../../../lib/env.fish

set -q CHECKPOINT_ROOT[1]
or set -lx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
set -q CONVERTED_PREFIX[1]
or set -lx CONVERTED_PREFIX dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
set -q HEIGHT[1]
or set -lx HEIGHT 512
set -q WIDTH[1]
or set -lx WIDTH 512
set -q OUTPUT_BASE[1]
or set -lx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028
set -q EVAL_LOG_DIR[1]
or set -lx EVAL_LOG_DIR storage/eval_logs/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_cps0p7_evalkit_4cc7d028

set -l script_dir (dirname (status filename))
fish $script_dir/../dancegrpo_manifest_rl_384x384x81_fujian/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_cps0p7_sweep_main_v2.fish $argv
exit $status
