#!/usr/bin/env fish
# Wan2.2 I2V Flow-GRPO training launcher
# Usage: fish scripts/train/grpo.fish [--nproc N] [training args...]
#   e.g. fish scripts/train/grpo.fish --config configs/train_grpo.yaml

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

set -l hsdp_backend_overridden false
for arg in $train_args
    if test "$arg" = "--hsdp_replicate_backend"
        set hsdp_backend_overridden true
        break
    end
    if string match -qr '^--hsdp_replicate_backend=' -- "$arg"
        set hsdp_backend_overridden true
        break
    end
end

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

# Keep local GRPO launches aligned with the multi-node launcher. Low memlock
# frequently forces NCCL/IB fallback, so default to NCCL over socket and log
# the effective transport/backend choice for quick diagnosis.
ulimit -l unlimited 2>/dev/null; or ulimit -l 67108864 2>/dev/null

set -l memlock_limit (ulimit -l 2>/dev/null)
set -l memlock_desc unknown
set -l low_memlock false
if test "$memlock_limit" = "unlimited"
    set memlock_desc unlimited
else if string match -qr '^[0-9]+$' -- "$memlock_limit"
    set memlock_desc "$memlock_limit KiB"
    if test "$memlock_limit" -le 8192
        set low_memlock true
    end
end

if test "$low_memlock" = true
    if not set -q WAN_NCCL_TRANSPORT
        set -gx WAN_NCCL_TRANSPORT socket
        echo "Detected low memlock ($memlock_desc); defaulting WAN_NCCL_TRANSPORT=socket"
    end
end

if not set -q NCCL_SOCKET_IFNAME
    set -gx NCCL_SOCKET_IFNAME eth0
end
if not set -q GLOO_SOCKET_IFNAME
    set -gx GLOO_SOCKET_IFNAME $NCCL_SOCKET_IFNAME
end
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
        if test "$hsdp_backend_overridden" = false
            set -a train_args --hsdp_replicate_backend nccl
        end
    end
end

set -l transport_desc ib
if set -q WAN_NCCL_TRANSPORT
    set transport_desc (string lower -- "$WAN_NCCL_TRANSPORT")
end
set -l hsdp_backend_desc auto
if test "$hsdp_backend_overridden" = true
    set hsdp_backend_desc user-specified
else if set -q WAN_NCCL_TRANSPORT
    set -l wan_nccl_transport (string lower -- "$WAN_NCCL_TRANSPORT")
    if contains -- $wan_nccl_transport socket tcp
        set hsdp_backend_desc nccl
    end
end

echo "Launching Flow-GRPO training with $nproc GPUs..."
echo "Distributed comm config: memlock=$memlock_desc transport=$transport_desc NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME hsdp_replicate_backend=$hsdp_backend_desc"
torchrun --nproc_per_node=$nproc -m src.cli.train_grpo $train_args
