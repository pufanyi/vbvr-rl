#!/usr/bin/env fish
# Wan2.2 I2V Flow-GRPO training launcher
# Usage: fish scripts/train/grpo.fish [--nproc N] [-- training args...]
#   e.g. fish scripts/train/grpo.fish --nproc 8 -- --config configs/train_grpo.yaml

set -l nproc 8

# Parse launcher args (before --)
set -l train_args
set -l parsing_launcher true
for arg in $argv
    if test "$arg" = "--"
        set parsing_launcher false
        continue
    end
    if $parsing_launcher
        if set -q _expect_nproc
            set nproc $arg
            set -e _expect_nproc
            continue
        end
        if test "$arg" = "--nproc"
            set -g _expect_nproc 1
            continue
        end
    else
        set -a train_args $arg
    end
end

# If no -- separator, treat all args as training args
if $parsing_launcher
    set train_args $argv
end

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

# Keep local GRPO launches aligned with the trainer defaults and allow the same
# `WAN_NCCL_TRANSPORT=socket` escape hatch used by the multi-node launcher.
if not set -q NCCL_REGISTRATION_CACHE_SIZE
    set -gx NCCL_REGISTRATION_CACHE_SIZE 0
end
if not set -q TORCH_NCCL_AVOID_RECORD_STREAMS
    set -gx TORCH_NCCL_AVOID_RECORD_STREAMS 1
end
if not set -q NCCL_NET_GDR_LEVEL
    set -gx NCCL_NET_GDR_LEVEL 0
end
if not set -q NCCL_IB_QPS_PER_CONNECTION
    set -gx NCCL_IB_QPS_PER_CONNECTION 1
end
if not set -q NCCL_IB_SPLIT_DATA_ON_QPS
    set -gx NCCL_IB_SPLIT_DATA_ON_QPS 0
end
if not set -q NCCL_MAX_NCHANNELS
    set -gx NCCL_MAX_NCHANNELS 4
end
if not set -q NCCL_MIN_NCHANNELS
    set -gx NCCL_MIN_NCHANNELS 1
end
if set -q WAN_NCCL_TRANSPORT
    set -l wan_nccl_transport (string lower -- "$WAN_NCCL_TRANSPORT")
    if contains -- $wan_nccl_transport socket tcp
        if not set -q NCCL_IB_DISABLE
            set -gx NCCL_IB_DISABLE 1
        end
    end
end

echo "Launching Flow-GRPO training with $nproc GPUs..."
torchrun --nproc_per_node=$nproc -m src.cli.train_grpo $train_args
