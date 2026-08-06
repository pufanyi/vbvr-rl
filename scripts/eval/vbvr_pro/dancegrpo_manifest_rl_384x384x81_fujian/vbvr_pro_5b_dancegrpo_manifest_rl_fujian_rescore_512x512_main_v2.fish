#!/usr/bin/env fish

# Score-only e140 migration for the native-512 evaluation of the Fujian
# manifest-RL checkpoints. The delegated launcher requires generation
# provenance for 512x512x81 videos and preparation provenance for the shared
# 1024x1024x81 scorer media; it never loads a checkpoint or uses a GPU.

source (dirname (status filename))/../../../lib/env.fish

set -lx SOURCE_OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_evalkit_eb977da6
set -lx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_rescore_from_evalkit_eb977da6_to_evalkit_4cc7d028
set -lx EVAL_LOG_DIR storage/eval_logs/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_rescore_evalkit_4cc7d028
set -lx EXPECTED_GENERATION_WIDTH 512
set -lx EXPECTED_GENERATION_HEIGHT 512

set -l script_dir (dirname (status filename))
fish $script_dir/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_rescore_main_v2.fish $argv
exit $status
