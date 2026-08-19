#!/usr/bin/env fish
# Compatibility entrypoint. The scale-out launcher supports 4/8/16 machines.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root
or exit 1

exec fish scripts/train/grpo_vlm_eval_scaleout.fish $argv
