#!/usr/bin/env fish

# Evaluate one checkpoint from the native-512 manifest-RL run. The shared
# evaluator owns dataset verification, conversion, generation, and scoring.

set -q CHECKPOINT_ROOT[1]
or set -lx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_evalkit_e140038f
if set -q CHECKPOINT_STEP[1]
    set -q CONVERTED_MODEL[1]
    or set -lx CONVERTED_MODEL storage/models/dcp_converted_5b/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_evalkit_e140038f_checkpoint-$CHECKPOINT_STEP
end
set -q HEIGHT[1]
or set -lx HEIGHT 512
set -q WIDTH[1]
or set -lx WIDTH 512
set -q OUTPUT_BASE[1]
or set -lx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028

set -l script_dir (dirname (status filename))
fish $script_dir/../dancegrpo_manifest_rl_384x384x81/vbvr_pro_5b_dancegrpo_manifest_rl_checkpoint_cps0p7_main_v2.fish $argv
exit $status
