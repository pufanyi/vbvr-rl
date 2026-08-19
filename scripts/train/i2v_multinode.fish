#!/usr/bin/env fish
# Wan2.2 I2V multi-node training launcher
#
# Expected environment variables (typically set by the job scheduler):
#   MASTER_ADDR  — hostname/IP of the master node
#   WORLD_SIZE   — number of nodes
#   RANK         — this node's rank (0-indexed)
#
# Optional environment variables:
#   MASTER_PORT  — port on master node (default: 29500)
#
# Usage: fish scripts/train/i2v_multinode.fish [--nproc N] [training args...]
#   e.g. fish scripts/train/i2v_multinode.fish --config configs/train_sft_vbvr_5e-6.yaml
#   e.g. fish scripts/train/i2v_multinode.fish --nproc 8 --config configs/train_sft_vbvr_5e-6.yaml

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

source (dirname (status filename))/../lib/env.fish

set -l master_port (set -q MASTER_PORT; and echo $MASTER_PORT; or echo 29500)

set -x NCCL_DEBUG INFO
set -x NCCL_DEBUG_SUBSYS INIT,NET,ENV

function __log_env_var
    set -l key $argv[1]
    if set -q $key
        echo "  $key="(string join " " $$key)
    else
        echo "  $key=<unset>"
    end
end

echo "Launching I2V multi-node training: node $RANK/$WORLD_SIZE, $nproc GPUs/node, master=$MASTER_ADDR:$master_port"
echo "NCCL/IB diagnostics before torchrun:"
for key in NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_IB_DISABLE NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_IB_GID_INDEX
    __log_env_var $key
end

if test -d /sys/class/infiniband
    set -l ib_devices (command ls -1 /sys/class/infiniband 2>/dev/null)
    if test (count $ib_devices) -gt 0
        echo "  /sys/class/infiniband: "(string join ", " $ib_devices)
    else
        echo "  /sys/class/infiniband: no devices"
    end
else
    echo "  /sys/class/infiniband: not present"
end

if type -q ibv_devinfo
    echo "ibv_devinfo:"
    ibv_devinfo
else if type -q ibstat
    echo "ibstat:"
    ibstat
else
    echo "  ibv_devinfo/ibstat: not found"
end

if type -q nvidia-smi
    echo "nvidia-smi topo -m:"
    nvidia-smi topo -m
end

echo "NCCL runtime network selection will be visible below; look for NET/IB or NET/Socket."

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$nproc \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$master_port \
    -m src.cli.train_i2v $train_args
