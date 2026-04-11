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

. .venv/bin/activate.fish

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$nproc \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$master_port \
    -m src.cli.train_grpo $train_args
