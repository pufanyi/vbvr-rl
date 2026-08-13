#!/usr/bin/env fish

# Scheduler-facing multi-node launcher for the complete 60-cell, 500-output,
# 30-step VBVR-Pro trajectory matrix.
#
# Expected scheduler environment (same node contract as grpo_multinode.fish):
#   WORLD_SIZE  — number of machines
#   RANK        — this machine's zero-based rank
#
# Usage on every 8-GPU machine:
#   WORLD_SIZE=8 RANK=0 fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/render_vbvr_pro_sampler_matrix_all_outputs_30steps_multinode.fish
#
# Optional:
#   --nproc N                       local GPUs to expose (default: 8)
#   --workers-per-gpu N             maximum independent B=1 workers/GPU (default: 2)
#   --sample-shards-per-cell N      deterministic sample shards/cell (default: workers/GPU)
#   --progress-interval N            live progress period in seconds (default: 30)
#   TRAJECTORY_ASSIGNMENT_ONLY=1    print this node's cells and exit
#   MODEL_FILTER / SAMPLER_FILTER   retain the base launcher's exact filters
#
# This is intentionally not torchrun: every model/sampler cell is an
# independent single-GPU inference job. The wrapper round-robin shards cells
# across nodes, while the base launcher fills the selected GPUs on each node.

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -l nproc 8
set -l workers_per_gpu 2
set -l sample_shards_per_cell
set -l progress_interval 30
set -l expect_nproc false
set -l expect_workers false
set -l expect_shards false
set -l expect_progress false
for arg in $argv
    if test "$expect_nproc" = true
        set nproc $arg
        set expect_nproc false
        continue
    end
    if test "$expect_workers" = true
        set workers_per_gpu $arg
        set expect_workers false
        continue
    end
    if test "$expect_shards" = true
        set sample_shards_per_cell $arg
        set expect_shards false
        continue
    end
    if test "$expect_progress" = true
        set progress_interval $arg
        set expect_progress false
        continue
    end
    if test "$arg" = "--nproc"
        set expect_nproc true
        continue
    end
    if test "$arg" = "--workers-per-gpu"
        set expect_workers true
        continue
    end
    if test "$arg" = "--sample-shards-per-cell"
        set expect_shards true
        continue
    end
    if test "$arg" = "--progress-interval"
        set expect_progress true
        continue
    end
    _fail "unknown argument '$arg' (supported: --nproc N --workers-per-gpu N --sample-shards-per-cell N --progress-interval N)"
end
test "$expect_nproc" = false; or _fail "--nproc requires a value"
test "$expect_workers" = false; or _fail "--workers-per-gpu requires a value"
test "$expect_shards" = false; or _fail "--sample-shards-per-cell requires a value"
test "$expect_progress" = false; or _fail "--progress-interval requires a value"
string match -qr '^[1-9][0-9]*$' -- "$nproc"
or _fail "--nproc must be a positive integer, got '$nproc'"
string match -qr '^[1-9][0-9]*$' -- "$workers_per_gpu"
or _fail "--workers-per-gpu must be a positive integer, got '$workers_per_gpu'"
if test -n "$sample_shards_per_cell"
    string match -qr '^[1-9][0-9]*$' -- "$sample_shards_per_cell"
    or _fail "--sample-shards-per-cell must be a positive integer, got '$sample_shards_per_cell'"
end
string match -qr '^[1-9][0-9]*$' -- "$progress_interval"
or _fail "--progress-interval must be a positive integer, got '$progress_interval'"

set -q WORLD_SIZE[1]; or _fail "WORLD_SIZE is not set"
set -q RANK[1]; or _fail "RANK is not set"
string match -qr '^[1-9][0-9]*$' -- "$WORLD_SIZE"
or _fail "WORLD_SIZE must be a positive integer, got '$WORLD_SIZE'"
string match -qr '^[0-9]+$' -- "$RANK"
or _fail "RANK must be a non-negative integer, got '$RANK'"
test $RANK -lt $WORLD_SIZE
or _fail "RANK=$RANK is outside [0, $WORLD_SIZE)"

set -l scheduler_world_size $WORLD_SIZE
set -l scheduler_rank $RANK
set -gx TRAJECTORY_NODE_COUNT $scheduler_world_size
set -gx TRAJECTORY_NODE_RANK $scheduler_rank
set -gx TRAJECTORY_LOCAL_GPU_COUNT $nproc
set -gx TRAJECTORY_WORKERS_PER_GPU $workers_per_gpu
if test -n "$sample_shards_per_cell"
    set -gx TRAJECTORY_SAMPLE_SHARDS_PER_CELL $sample_shards_per_cell
end
set -gx TRAJECTORY_PROGRESS_INTERVAL $progress_interval
set -e TRAJECTORY_CUDA_DEVICES
set -l local_devices (seq 0 (math $nproc - 1))

set -l script_dir (realpath (dirname (status filename)))
set -l base_launcher $script_dir/render_vbvr_pro_sampler_matrix_all_outputs_30steps.fish
test -f $base_launcher; or _fail "base launcher is missing: $base_launcher"

set -l shard_text (set -q TRAJECTORY_SAMPLE_SHARDS_PER_CELL[1]; and echo $TRAJECTORY_SAMPLE_SHARDS_PER_CELL; or echo default)
echo "[multinode] node=$scheduler_rank/$scheduler_world_size local_gpus=$nproc workers_per_gpu=$workers_per_gpu sample_shards_per_cell=$shard_text"
echo "[multinode] devices=$local_devices"
echo "[multinode] cells are round-robin sharded by node; no cross-node process group is created"

# RANK/WORLD_SIZE describe scheduler nodes here, not PyTorch worker ranks.
# Prevent Diffusers/model code in the independent single-GPU children from
# seeing an ambient distributed launch contract.
set -e RANK
set -e WORLD_SIZE
set -e LOCAL_RANK
set -e LOCAL_WORLD_SIZE
set -e NODE_RANK
set -e GROUP_RANK
set -e ROLE_RANK
set -e ROLE_WORLD_SIZE
set -e MASTER_ADDR
set -e MASTER_PORT

exec fish $base_launcher
