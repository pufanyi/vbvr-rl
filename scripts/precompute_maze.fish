#!/usr/bin/env fish

source (dirname (status filename))/activate_venv.fish

set -x PYTHONPATH (pwd) $PYTHONPATH

torchrun --nproc_per_node=8 scripts/precompute_latents.py \
    --input data/train_maze_bfs_sft.json \
    --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
    --output_dir data/maze \
    --batch_size 4
