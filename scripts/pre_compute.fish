#!/usr/bin/env fish

source (dirname (status filename))/activate_venv.fish

set -x PYTHONPATH (pwd)

torchrun \
        --nnodes=$WORLD_SIZE --nproc_per_node=1 \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        scripts/precompute_vbvr_latents.py \
        --metadata data/vbvr/VBVR-Dataset/data/metadata.parquet \
        --tar_dir data/vbvr/VBVR-Dataset/tars \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --output_dir /shared/vbvr/latents \
        --batch_size 4
