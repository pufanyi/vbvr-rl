#!/usr/bin/env fish
# Wait for the Maze perfect-v2 SFT WebDataset split, then run one COS epoch.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root; or exit 1

source scripts/lib/env.fish

set -q NPROC; or set NPROC 8
set -q EXPECTED_SHARDS; or set EXPECTED_SHARDS 80
set -q WAIT_SECONDS; or set WAIT_SECONDS 60

set -l data_dir data/maze/latents/maze_384x384x81_perfect_v2/webdataset/sft
set -l config configs/train_cos_maze_perfect_v2_sft_1epoch.yaml

while true
    set -l ready (find $data_dir -maxdepth 1 -type f -name 'shard-*.tar' | wc -l | string trim)
    set -l tmp (find $data_dir -maxdepth 1 -type f -name 'shard-*.tar.tmp' | wc -l | string trim)

    if test $ready -ge $EXPECTED_SHARDS; and test $tmp -eq 0
        echo "[ready] $data_dir has $ready shard-*.tar files and no tmp shards"
        break
    end

    echo "[wait] $data_dir ready=$ready/$EXPECTED_SHARDS tmp=$tmp; sleeping {$WAIT_SECONDS}s"
    sleep $WAIT_SECONDS
end

exec fish scripts/train/i2v.fish --nproc $NPROC -- --config $config
