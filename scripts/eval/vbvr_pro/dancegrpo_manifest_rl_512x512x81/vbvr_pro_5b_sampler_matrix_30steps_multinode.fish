#!/usr/bin/env fish

# Scheduler-facing wrapper for the native-512 quantitative sampler matrix.
# WORLD_SIZE is the machine count and RANK is this machine's zero-based rank.
# Every formal cell uses two local GPUs, so eight-GPU nodes run four cells at
# once. This wrapper only partitions independent cells; it creates no torchrun
# process group across machines.

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -l nproc 8
set -l checkpoints
set -l include_baseline 1
set -l assignment_only 0
set -l expect_value
for arg in $argv
    if test -n "$expect_value"
        switch $expect_value
            case nproc
                set nproc $arg
            case checkpoints
                set checkpoints $arg
        end
        set expect_value
        continue
    end
    switch $arg
        case --nproc
            set expect_value nproc
        case --checkpoints
            set expect_value checkpoints
        case --no-baseline
            set include_baseline 0
        case --include-baseline
            set include_baseline 1
        case --assignment-only
            set assignment_only 1
        case '*'
            _fail "unknown argument '$arg'"
    end
end
test -z "$expect_value"; or _fail "--$expect_value requires a value"
string match -qr '^[1-9][0-9]*$' -- "$nproc"
or _fail "--nproc must be a positive integer: $nproc"
test (math "$nproc % 2") -eq 0
or _fail "--nproc must be even because each formal cell uses two GPUs"

set -q WORLD_SIZE[1]; or _fail "WORLD_SIZE is not set"
set -q RANK[1]; or _fail "RANK is not set"
string match -qr '^[1-9][0-9]*$' -- "$WORLD_SIZE"
or _fail "WORLD_SIZE must be a positive integer: $WORLD_SIZE"
string match -qr '^[0-9]+$' -- "$RANK"
or _fail "RANK must be a non-negative integer: $RANK"
test $RANK -lt $WORLD_SIZE
or _fail "RANK=$RANK is outside [0, $WORLD_SIZE)"

set -l scheduler_world_size $WORLD_SIZE
set -l scheduler_rank $RANK
set -gx MATRIX_NODE_COUNT $scheduler_world_size
set -gx MATRIX_NODE_RANK $scheduler_rank
set -gx MATRIX_LOCAL_GPU_COUNT $nproc
set -gx MATRIX_INCLUDE_BASELINE $include_baseline
if test -n "$checkpoints"
    set -gx MATRIX_CHECKPOINT_STEPS $checkpoints
end
if test $assignment_only -eq 1
    set -gx MATRIX_ASSIGNMENT_ONLY 1
end

set -l script_dir (realpath (dirname (status filename)))
set -l base_launcher $script_dir/vbvr_pro_5b_sampler_matrix_30steps_main_v2.fish
test -f $base_launcher; or _fail "base launcher is missing: $base_launcher"

echo "[multinode-matrix] node=$scheduler_rank/$scheduler_world_size local_gpus=$nproc"
echo "[multinode-matrix] checkpoints="(set -q MATRIX_CHECKPOINT_STEPS[1]; and echo $MATRIX_CHECKPOINT_STEPS; or echo auto)
echo "[multinode-matrix] include_baseline=$include_baseline"

# These variables describe scheduler nodes here. The per-cell launcher creates
# its own local two-GPU process group and must not inherit the outer contract.
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
