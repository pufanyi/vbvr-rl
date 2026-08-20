#!/usr/bin/env fish
# Wan2.2 I2V DanceGRPO training launcher
# Usage: fish scripts/train/grpo.fish [--nproc N] [training args...]
#   e.g. fish scripts/train/grpo.fish --nproc 8 --config configs/train_rl_a14b_rule.yaml

set -l nproc 8

# Parse launcher args. Unknown args are forwarded to the training entrypoint,
# so `fish ... --config=...` works without an explicit `--` separator.
set -l train_args
set -l parsing_launcher true
set -l expect_nproc false
for arg in $argv
    if test "$arg" = "--"
        set parsing_launcher false
        continue
    end

    if test "$expect_nproc" = true
        set nproc $arg
        set expect_nproc false
        continue
    end

    if $parsing_launcher
        if test "$arg" = "--nproc"
            set expect_nproc true
            continue
        end
    end

    set -a train_args $arg
end

if test "$expect_nproc" = true
    echo "ERROR: --nproc requires a value" >&2
    exit 1
end

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

source (dirname (status filename))/../lib/env.fish

# Long A14B replay repeatedly all-gathers large FSDP blocks. Expandable CUDA
# segments reduce allocator fragmentation; preserve an explicit operator
# override when one is already set.
set -q PYTORCH_CUDA_ALLOC_CONF; or set -gx PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
set -q WAN_TRAINER_DECORD_NUM_THREADS; or set -gx WAN_TRAINER_DECORD_NUM_THREADS 1
set -q TRITON_CACHE_DIR; or set -gx TRITON_CACHE_DIR /tmp/wan-trainer-triton-cache
# Hub attention bundles are large and must survive scheduler job turnover.
# Training loads the predownloaded pinned snapshot offline; use
# src.cli.prefetch_attention_kernel once on a networked login node.
if set -q WAN_TRAINER_KERNELS_CACHE; and test -n "$WAN_TRAINER_KERNELS_CACHE"
    set -gx KERNELS_CACHE $WAN_TRAINER_KERNELS_CACHE
else
    # Do not preserve an ambient KERNELS_CACHE: scheduler images may inject an
    # ephemeral /tmp path that disappears between jobs.
    set -gx KERNELS_CACHE ~/.cache/wan-trainer/kernels
end
echo "[preflight] Persistent attention kernel cache: $KERNELS_CACHE"

pixi run --locked python -m src.cli.validate_grpo_runtime $train_args
or begin
    echo "ERROR: GRPO runtime preflight failed before torchrun." >&2
    exit 1
end

echo "Launching DanceGRPO training with $nproc GPUs..."

torchrun --nproc_per_node=$nproc -m src.cli.train_grpo $train_args
