#!/usr/bin/env fish
# Wan2.2 I2V Flow-GRPO training launcher
# Usage: fish scripts/train/grpo.fish [--nproc N] [training args...]
#   e.g. fish scripts/train/grpo.fish --config configs/train_grpo.yaml

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

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

source (dirname (status filename))/../lib/env.fish

echo "Launching Flow-GRPO training with $nproc GPUs..."

torchrun --nproc_per_node=$nproc -m src.cli.train_grpo $train_args
