#!/usr/bin/env fish
# Compatibility entrypoint. The cluster launcher now detects 4/8/16 nodes.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root
or exit 1

exec fish scripts/train/grpo_vlm_eval_cluster.fish $argv
