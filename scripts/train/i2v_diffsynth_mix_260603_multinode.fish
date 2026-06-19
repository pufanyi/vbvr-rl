#!/usr/bin/env fish
# Wan2.2 A14B DiffSynth mix raw I2V multi-node training launcher.
#
# Expected environment variables, usually provided by the scheduler:
#   MASTER_ADDR  - hostname/IP of the master node
#   WORLD_SIZE   - number of nodes
#   RANK         - this node's rank, 0-indexed
#
# Optional environment variables:
#   MASTER_PORT  - port on master node (default: 29500)
#
# Usage:
#   fish scripts/train/i2v_diffsynth_mix_260603_multinode.fish --nproc 8
#   fish scripts/train/i2v_diffsynth_mix_260603_multinode.fish --nproc 8 --config configs/train_sft_diffsynth_mix_260603_smoke.yaml
#   fish scripts/train/i2v_diffsynth_mix_260603_multinode.fish --nproc 8 -- --max_steps 100

set -l nproc 8
set -l default_config configs/train_sft_diffsynth_mix_260603.yaml

# Parse launcher args. Unknown args are forwarded to the training entrypoint.
# A `--` separator is accepted but not required.
set -l train_args
set -l parsing_launcher true
set -l expect_nproc false
set -l saw_config false

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

    if test "$arg" = "--config"; or string match -q -- "--config=*" "$arg"
        set saw_config true
    end

    set -a train_args $arg
end

if test "$expect_nproc" = true
    echo "ERROR: --nproc requires a value" >&2
    exit 1
end

if not $saw_config
    set train_args --config $default_config $train_args
end

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

set -q MASTER_PORT; or set -gx MASTER_PORT 29500
set -q WAN_TRAINER_AOSS_CONF_RULES; or set -gx WAN_TRAINER_AOSS_CONF_RULES '[{"pattern":"^s3://multimodal","conf_path":"/mnt/aigc/users/pufanyi/aoss.conf"}]'
set -q WAN_TRAINER_AOSS_CONF_PATH; or set -gx WAN_TRAINER_AOSS_CONF_PATH /mnt/aigc/caizhongang/aoss.conf
set -q WAN_TRAINER_REMOTE_CACHE_DIR; or set -gx WAN_TRAINER_REMOTE_CACHE_DIR storage/aoss_cache
set -q WAN_TRAINER_DECORD_NUM_THREADS; or set -gx WAN_TRAINER_DECORD_NUM_THREADS 1
set -q PYTORCH_CUDA_ALLOC_CONF; or set -gx PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
set -q LOGURU_LEVEL; or set -gx LOGURU_LEVEL INFO

if test -d $WAN_TRAINER_ROOT/aoss
    set -gx PYTHONPATH $WAN_TRAINER_ROOT $WAN_TRAINER_ROOT/aoss $PYTHONPATH
end

set -q NCCL_DEBUG; or set -gx NCCL_DEBUG WARN

function __log_env_var
    set -l key $argv[1]
    if set -q $key
        echo "  $key="(string join -- " " $$key)
    else
        echo "  $key=<unset>"
    end
end

echo "Launching DiffSynth mix I2V raw training: node $RANK/$WORLD_SIZE, $nproc GPUs/node, master=$MASTER_ADDR:$MASTER_PORT"
echo "Training args: "(string join -- " " $train_args)
echo "AOSS/cache:"
for key in WAN_TRAINER_AOSS_CONF_RULES WAN_TRAINER_AOSS_CONF_PATH WAN_TRAINER_REMOTE_CACHE_DIR WAN_TRAINER_DECORD_NUM_THREADS PYTORCH_CUDA_ALLOC_CONF
    __log_env_var $key
end
echo "NCCL/IB diagnostics before torchrun:"
for key in NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_IB_DISABLE NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_IB_GID_INDEX
    __log_env_var $key
end

if test -d /sys/class/infiniband
    set -l ib_devices (command ls -1 /sys/class/infiniband 2>/dev/null)
    if test (count $ib_devices) -gt 0
        echo "  /sys/class/infiniband: "(string join -- ", " $ib_devices)
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

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$nproc \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m src.cli.train_i2v $train_args

sleep 3h