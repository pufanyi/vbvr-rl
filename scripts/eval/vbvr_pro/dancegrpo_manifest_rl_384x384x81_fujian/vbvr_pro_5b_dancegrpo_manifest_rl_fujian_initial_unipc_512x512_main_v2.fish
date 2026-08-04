#!/usr/bin/env fish

# Fixed native-512 entry point for the DiffSynth step-35500 initialization.
# The delegated launcher keeps 50-step UniPC ODE, CFG 5.0, seed 0, 81 frames,
# exact 16 FPS, the pinned public snapshot, and the pinned scorer contract.

source (dirname (status filename))/../../../lib/env.fish

set -lx HEIGHT 512
set -lx WIDTH 512
set -lx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028

set -l script_dir (dirname (status filename))
fish $script_dir/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_initial_unipc_main_v2.fish $argv
exit $status
