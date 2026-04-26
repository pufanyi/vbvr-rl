#!/usr/bin/env fish
# Build WebDataset tar shards from precomputed prompt embeds + VAE latents
#
# Usage:
#   fish scripts/data/vbvr_build_webdataset.fish

set -q PROMPT_EMBEDS_DIR; or set -gx PROMPT_EMBEDS_DIR data/vbvr/latents/prompt_embeds
set -q VAE_LATENTS_DIR;   or set -gx VAE_LATENTS_DIR   data/vbvr/latents/vae_latents
set -q OUTPUT_DIR;         or set -gx OUTPUT_DIR         data/vbvr/latents/webdataset
set -q SAMPLES_PER_SHARD;  or set -gx SAMPLES_PER_SHARD  1000
set -q SEED;               or set -gx SEED               42

source (dirname (status filename))/../lib/env.fish

python -m src.precompute.build_webdataset \
    --prompt_embeds_dir $PROMPT_EMBEDS_DIR \
    --vae_latents_dir $VAE_LATENTS_DIR \
    --output_dir $OUTPUT_DIR \
    --samples_per_shard $SAMPLES_PER_SHARD \
    --seed $SEED
