#!/usr/bin/env fish
# Wan2.2 I2V DanceGRPO split-RL multi-node launcher.
#
# Layout:
#   WORLD_SIZE=4 nodes => nodes 0-1 train, nodes 2-3 run rollout/reward actors.
#   WORLD_SIZE=8 nodes => nodes 0-3 train, nodes 4-7 run rollout/reward actors.
#   Default split is half training nodes and half rollout/reward actor nodes.
#   --train-nodes N => first N nodes train, remaining nodes run rollout/reward actors.
#   WORLD_SIZE=1 node + --train-ranks N => first N local ranks train, remaining ranks
#     run rollout/reward actors. This is intended for single-node smoke tests.
#
# Expected scheduler environment:
#   MASTER_ADDR  — hostname/IP of the master node
#   WORLD_SIZE   — number of nodes
#   RANK         — this node's rank (0-indexed)
#
# Usage:
#   fish scripts/train/dancegrpo_maze_split_multinode.fish --nproc 8 \
#       --config configs/<reviewed-split-rl-config>.yaml
#   fish scripts/train/dancegrpo_maze_split_multinode.fish --nproc 8 \
#       --train-nodes 2 --config configs/<reviewed-split-rl-config>.yaml
#   WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 \
#       fish scripts/train/dancegrpo_maze_split_multinode.fish --nproc 8 \
#       --train-ranks 4 --config configs/<split-rl-config>.yaml

set -l nproc 8
set -l train_node_count auto
set -l train_rank_count 0

set -l train_args
set -l parsing_launcher true
set -l expect_nproc false
set -l expect_train_nodes false
set -l expect_train_ranks false
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

    if test "$expect_train_nodes" = true
        set train_node_count $arg
        set expect_train_nodes false
        continue
    end

    if test "$expect_train_ranks" = true
        set train_rank_count $arg
        set expect_train_ranks false
        continue
    end

    if $parsing_launcher
        if test "$arg" = "--nproc"
            set expect_nproc true
            continue
        end
        if test "$arg" = "--train-nodes"
            set expect_train_nodes true
            continue
        end
        if string match -q -- "--train-nodes=*" "$arg"
            set train_node_count (string replace -- "--train-nodes=" "" "$arg")
            continue
        end
        if test "$arg" = "--train-ranks"
            set expect_train_ranks true
            continue
        end
        if string match -q -- "--train-ranks=*" "$arg"
            set train_rank_count (string replace -- "--train-ranks=" "" "$arg")
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
if test "$expect_train_nodes" = true
    echo "ERROR: --train-nodes requires a value" >&2
    exit 1
end
if test "$expect_train_ranks" = true
    echo "ERROR: --train-ranks requires a value" >&2
    exit 1
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

if not string match -qr '^[1-9][0-9]*$' -- "$nproc"
    echo "ERROR: --nproc must be a positive integer, got $nproc" >&2
    exit 1
end
if test "$train_node_count" != "auto"; and not string match -qr '^[1-9][0-9]*$' -- "$train_node_count"
    echo "ERROR: --train-nodes must be a positive integer or auto, got $train_node_count" >&2
    exit 1
end
if not string match -qr '^[0-9]+$' -- "$train_rank_count"
    echo "ERROR: --train-ranks must be a non-negative integer, got $train_rank_count" >&2
    exit 1
end

if test "$saw_config" = false
    echo "ERROR: --config is required; choose a reviewed split-RL config" >&2
    exit 1
end

set -l master_port (set -q MASTER_PORT; and echo $MASTER_PORT; or echo 29500)
set -l total_ranks (math "$WORLD_SIZE * $nproc")
set -l split_args
set -l split_desc

if test "$train_rank_count" -gt 0
    if test "$train_rank_count" -ge "$total_ranks"
        echo "ERROR: --train-ranks must leave at least one rollout rank; got train_ranks=$train_rank_count total_ranks=$total_ranks" >&2
        exit 1
    end
    set split_args --rl_train_node_count 0 --rl_train_rank_count $train_rank_count
    set split_desc "train_ranks=$train_rank_count, inference_ranks="(math "$total_ranks - $train_rank_count")
else
    if test "$WORLD_SIZE" != "4"; and test "$WORLD_SIZE" != "8"
        echo "ERROR: this launcher is intended for WORLD_SIZE=4 or WORLD_SIZE=8 nodes unless --train-ranks is set, got $WORLD_SIZE" >&2
        exit 1
    end
    if test "$train_node_count" = "auto"
        set train_node_count (math "$WORLD_SIZE / 2")
    end
    if test "$train_node_count" -ge "$WORLD_SIZE"
        echo "ERROR: --train-nodes must leave at least one rollout node; got train_nodes=$train_node_count total_nodes=$WORLD_SIZE" >&2
        exit 1
    end
    set -l infer_nodes (math "$WORLD_SIZE - $train_node_count")
    set split_args --rl_train_node_count $train_node_count --rl_train_rank_count 0
    set split_desc "train_nodes=$train_node_count, inference_nodes=$infer_nodes"
end

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

source (dirname (status filename))/../lib/env.fish

echo "Launching DanceGRPO split RL: node $RANK/$WORLD_SIZE, $nproc GPUs/node, $split_desc, master=$MASTER_ADDR:$master_port"

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$nproc \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$master_port \
    -m src.cli.train_grpo \
    --trainer dancegrpo \
    $split_args \
    $train_args
