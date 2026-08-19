#!/usr/bin/env fish
# Four-node production wrapper for full-FT A14B VBVR-Pro DanceGRPO.
#
# The base config is the validated one-node TP2 x FSDP4 setup, where
# batch_size=4 over DP4 gives 16 global prompts. Four nodes create DP16, so
# this wrapper fixes the local batch at 1 and isolates checkpoints/logging under
# an explicit TP2 x FSDP16 four-node run name.
#
# Expected scheduler environment: MASTER_ADDR, WORLD_SIZE=4, RANK=0..3.
# Extra arguments are forwarded last and may override these defaults.

if not set -q WORLD_SIZE; or test "$WORLD_SIZE" != "4"
    echo "ERROR: this launcher requires WORLD_SIZE=4 nodes, got '$WORLD_SIZE'" >&2
    exit 1
end

set -l script_dir (realpath (dirname (status filename)))
set -l config configs/train_rl_a14b_rule.yaml
set -l run_stem dancegrpo_vbvr_pro_a14b_256x256x161_rule_cps_from_sft_diffsynth_mix_260603_bs16_lr_1e-5_full_tp2_fsdp16_4node_liger_compile

exec fish $script_dir/grpo_multinode.fish \
    --nproc 8 \
    --config $config \
    --batch_size 1 \
    --output_dir storage/checkpoints/$run_stem \
    --wandb_run_name $run_stem \
    --vbvr_reward_tmp_dir storage/tmp/vbvr_pro_a14b_rule_reward_tp2_fsdp16_4node \
    $argv
