#!/usr/bin/env fish
# Wan2.2 I2V multi-node training launcher
#
# Expected environment variables (typically set by the cluster scheduler):
#   MASTER_ADDR  — hostname/IP of the master node
#   WORLD_SIZE   — number of nodes
#   RANK         — this node's rank (0-indexed)
#
# Optional environment variables:
#   MASTER_PORT  — port on master node (default: 29500)
#
# Usage: fish scripts/train/i2v_multinode.fish [--nproc N] [training args...]
#   e.g. fish scripts/train/i2v_multinode.fish --config configs/train_i2v.yaml
#   e.g. fish scripts/train/i2v_multinode.fish --nproc 8 --config configs/train_i2v.yaml

set -l nproc 8

# Parse launcher args. Unknown args are forwarded to the training entrypoint,
# so `fish ... --config=...` works without an explicit `--` separator.
set -l train_args
set -l expect_nproc false
for arg in $argv
    if test "$expect_nproc" = true
        set nproc $arg
        set expect_nproc false
        continue
    end

    if test "$arg" = "--nproc"
        set expect_nproc true
        continue
    end

    set -a train_args $arg
end

if test "$expect_nproc" = true
    echo "ERROR: --nproc requires a value" >&2
    exit 1
end

# # Validate required environment variables
# if not set -q MASTER_ADDR; or test -z "$MASTER_ADDR"
#     echo "ERROR: MASTER_ADDR is not set" >&2
#     exit 1
# end
# if not set -q WORLD_SIZE; or test -z "$WORLD_SIZE"
#     echo "ERROR: WORLD_SIZE is not set" >&2
#     exit 1
# end
# if not set -q RANK; or test -z "$RANK"
#     echo "ERROR: RANK is not set" >&2
#     exit 1
# end

# set -l project_root (realpath (dirname (status filename))/../..)
# cd $project_root

# echo "Launching multi-node training: node $RANK/$WORLD_SIZE, $nproc GPUs/node, master=$MASTER_ADDR:$master_port"

# # Try to raise memlock for NCCL/IB. This often fails under schedulers; we
# # still continue and fall back to NCCL over socket when the limit is too low.
# ulimit -l unlimited 2>/dev/null; or ulimit -l 67108864 2>/dev/null

# set -l memlock_limit (ulimit -l 2>/dev/null)
# set -l memlock_desc unknown
# set -l low_memlock false
# if test "$memlock_limit" = "unlimited"
#     set memlock_desc unlimited
# else if string match -qr '^[0-9]+$' -- "$memlock_limit"
#     set memlock_desc "$memlock_limit KiB"
#     if test "$memlock_limit" -le 8192
#         set low_memlock true
#     end
# end

# # Low memlock makes NCCL/IB registration fragile. Keep NCCL, but default the
# # transport to socket/TCP instead of trying to run the hot path through gloo.
# if test "$low_memlock" = true
#     if not set -q WAN_NCCL_TRANSPORT
#         set -gx WAN_NCCL_TRANSPORT socket
#         echo "Detected low memlock ($memlock_desc); defaulting WAN_NCCL_TRANSPORT=socket"
#     end
# end

# # Default bootstrap interfaces. Override from the environment if your cluster
# # uses a different NIC name.
# if not set -q NCCL_SOCKET_IFNAME
#     set -gx NCCL_SOCKET_IFNAME eth0
# end
# if not set -q GLOO_SOCKET_IFNAME
#     set -gx GLOO_SOCKET_IFNAME $NCCL_SOCKET_IFNAME
# end

# set -l transport_desc ib
# if set -q WAN_NCCL_TRANSPORT
#     set -l wan_nccl_transport (string lower -- "$WAN_NCCL_TRANSPORT")
#     set transport_desc $wan_nccl_transport
#     if contains -- $wan_nccl_transport socket tcp
#         if not set -q NCCL_IB_DISABLE
#             set -gx NCCL_IB_DISABLE 1
#         end
#     end
# end

# set -l ib_disable_desc 0
# if set -q NCCL_IB_DISABLE
#     set ib_disable_desc $NCCL_IB_DISABLE
# end
# echo "Distributed comm config: memlock=$memlock_desc transport=$transport_desc NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME NCCL_IB_DISABLE=$ib_disable_desc"

export NCCL_IB_HCA=mlx5_0:1
export NCCL_IB_DISABLE=0 
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=23
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^

. .venv/bin/activate.fish

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$nproc \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m src.cli.train_i2v $train_args
