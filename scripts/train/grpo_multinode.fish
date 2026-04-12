#!/usr/bin/env fish
# Wan2.2 I2V Flow-GRPO multi-node training launcher
#
# Expected environment variables (typically set by the cluster scheduler):
#   MASTER_ADDR  — hostname/IP of the master node
#   WORLD_SIZE   — number of nodes
#   RANK         — this node's rank (0-indexed)
#
# Optional environment variables:
#   MASTER_PORT  — port on master node (default: 29500)
#
# Usage: fish scripts/train/grpo_multinode.fish [--nproc N] [training args...]
#   e.g. fish scripts/train/grpo_multinode.fish --config configs/train_grpo.yaml
#   e.g. fish scripts/train/grpo_multinode.fish --nproc 8 --config configs/train_grpo.yaml

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

# Validate required environment variables
if not set -q MASTER_ADDR; or test -z "$MASTER_ADDR"
    echo "ERROR: MASTER_ADDR is not set" >&2
    exit 1
end
if not set -q WORLD_SIZE; or test -z "$WORLD_SIZE"
    echo "ERROR: WORLD_SIZE is not set" >&2
    exit 1
end
if not set -q RANK; or test -z "$RANK"
    echo "ERROR: RANK is not set" >&2
    exit 1
end

set -l master_port (set -q MASTER_PORT; and echo $MASTER_PORT; or echo 29500)

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

echo "Launching Flow-GRPO multi-node training: node $RANK/$WORLD_SIZE, $nproc GPUs/node, master=$MASTER_ADDR:$master_port"

# Ensure sufficient locked memory for InfiniBand RDMA registration
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

# Low memlock makes NCCL/IB registration fragile. Default to NCCL over socket
# so HSDP cross-node replicate traffic can stay on GPU collectives instead of
# falling back to the much slower gloo backend.
if test "$low_memlock" = true
    if not set -q WAN_NCCL_TRANSPORT
        set -gx WAN_NCCL_TRANSPORT socket
        echo "Detected low memlock ($memlock_desc); defaulting WAN_NCCL_TRANSPORT=socket"
    end
end

# NCCL tuning for multi-node FSDP with large all_gather buffers. Preserve any
# user-provided overrides and support `WAN_NCCL_TRANSPORT=socket` as a hard
# fallback when InfiniBand queue-pair allocation is exhausted.
if not set -q NCCL_IB_GID_INDEX
    set -gx NCCL_IB_GID_INDEX 3
end
if not set -q NCCL_IB_RETRY_CNT
    set -gx NCCL_IB_RETRY_CNT 7
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
echo "Distributed comm config: memlock=$memlock_desc transport=$transport_desc NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME hsdp_replicate_backend=$hsdp_backend_desc"

. .venv/bin/activate.fish

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$nproc \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$master_port \
    -m src.cli.train_grpo $train_args
