#!/usr/bin/env fish
# Generate a synthetic maze WebDataset with precomputed Wan2.2 latents.
#
# Single node (auto-detects GPU count):
#   ./scripts/gen_maze_webdataset.fish --num_samples 20000
#
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT and run the
# script on every node. Each node uses all local GPUs via NPROC (default
# `nvidia-smi -L | wc -l`). Shards are split round-robin across global ranks,
# so you can just let all ranks write to the same --output_dir:
#
#   # on rank-0 node:
#   env NNODES=2 NODE_RANK=0 MASTER_ADDR=node0.internal MASTER_PORT=29500 \
#       ./scripts/gen_maze_webdataset.fish --num_samples 20000
#   # on rank-1 node (same MASTER_ADDR / MASTER_PORT):
#   env NNODES=2 NODE_RANK=1 MASTER_ADDR=node0.internal MASTER_PORT=29500 \
#       ./scripts/gen_maze_webdataset.fish --num_samples 20000
#
# Output directory is consumed by `latent_webdataset_dir:` in the training
# YAML (same contract as VBVRLatentDataset).

source (dirname (status filename))/activate_venv.fish

set -x PYTHONPATH (pwd) $PYTHONPATH

set -l n_gpus (nvidia-smi -L | wc -l)
if test $n_gpus -lt 1
    set n_gpus 1
end

set -q NPROC; or set NPROC $n_gpus
set -q NNODES; or set NNODES 1
set -q NODE_RANK; or set NODE_RANK 0
set -q MASTER_ADDR; or set MASTER_ADDR 127.0.0.1
set -q MASTER_PORT; or set MASTER_PORT 29500

echo "[gen_maze_webdataset] nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NPROC master=$MASTER_ADDR:$MASTER_PORT"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m src.precompute.maze_webdataset \
    --output_dir data/maze_synth/latents/webdataset \
    --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
    --num_samples 20000 \
    --samples_per_shard 500 \
    --cell_h 6 \
    --cell_w 10 \
    --cell_px 32 \
    --num_frames 81 \
    --vae_batch_size 4 \
    --text_batch_size 64 \
    --seed 42 \
    --skip_existing \
    $argv
