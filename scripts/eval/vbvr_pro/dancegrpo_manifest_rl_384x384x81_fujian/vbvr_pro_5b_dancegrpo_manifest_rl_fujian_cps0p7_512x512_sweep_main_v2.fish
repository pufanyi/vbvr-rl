#!/usr/bin/env fish

# Fixed native-512 inference sweep for all complete checkpoints in the Fujian
# 384x384x81 training run. The delegated sweep keeps 30-step Flow-CPS 0.7,
# CFG 1.0, seed 0, 81 frames, and exact 16-FPS playback.

source (dirname (status filename))/../../../lib/env.fish

set -lx HEIGHT 512
set -lx WIDTH 512
set -lx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028
set -lx EVAL_LOG_DIR storage/eval_logs/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_cps0p7_evalkit_4cc7d028

set -l script_dir (dirname (status filename))
fish $script_dir/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_cps0p7_sweep_main_v2.fish $argv
exit $status
