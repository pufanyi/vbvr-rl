#!/usr/bin/env fish

# Eight-machine evaluation for the immutable 2026-08-07 checkpoint extension:
#   checkpoint-1000, 1100, 1200, 1300, and 1400
#
# Run `formal` on every node first. After all eight nodes exit successfully,
# run `trajectories` on every node to render all 500 clean 30-step galleries.
# Existing baseline/checkpoint-100..900 cells are intentionally excluded.

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -l stage
set -l forwarded
set -l assignment_only 0
for arg in $argv
    switch $arg
        case formal trajectories
            test -z "$stage"; or _fail "specify exactly one stage: formal or trajectories"
            set stage $arg
        case --assignment-only
            set assignment_only 1
        case '*'
            set -a forwarded $arg
    end
end
test -n "$stage"; or set stage formal

set -q NEW_CHECKPOINT_STEPS[1]
or set -g NEW_CHECKPOINT_STEPS 1000 1100 1200 1300 1400
set -l steps
set -l requested_steps (string match -ra '[^,[:space:]]+' -- (string join ',' -- $NEW_CHECKPOINT_STEPS))
for step in $requested_steps
    string match -qr '^[1-9][0-9]*$' -- "$step"
    or _fail "NEW_CHECKPOINT_STEPS contains an invalid step: $step"
    contains -- $step $steps; and _fail "NEW_CHECKPOINT_STEPS contains duplicate step $step"
    set -a steps $step
end
test (count $steps) -gt 0; or _fail "NEW_CHECKPOINT_STEPS must not be empty"

set -l checkpoint_root storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
for step in $steps
    test -f $checkpoint_root/checkpoint-$step/high/.metadata
    or _fail "checkpoint-$step is missing or incomplete under $checkpoint_root"
end

# Export one scalar so the child Fish process receives the exact same snapshot
# on every scheduler node. Both base launchers normalize comma-separated steps.
set -gx MATRIX_CHECKPOINT_STEPS (string join ',' -- $steps)
set -gx MATRIX_INCLUDE_BASELINE 0
if test $assignment_only -eq 0
    set -e MATRIX_ASSIGNMENT_ONLY
    set -e TRAJECTORY_ASSIGNMENT_ONLY
end

set -l script_dir (realpath (dirname (status filename)))
switch $stage
    case formal
        set -l launcher $script_dir/vbvr_pro_5b_sampler_matrix_30steps_multinode.fish
        set -l stage_args --checkpoints $MATRIX_CHECKPOINT_STEPS --no-baseline $forwarded
        if test $assignment_only -eq 1
            set -a stage_args --assignment-only
        end
        echo "[new-checkpoints] stage=formal steps=$steps"
        exec fish $launcher $stage_args
    case trajectories
        set -l launcher $script_dir/render_vbvr_pro_sampler_matrix_all_outputs_30steps_multinode.fish
        if test $assignment_only -eq 1
            set -gx TRAJECTORY_ASSIGNMENT_ONLY 1
        end
        echo "[new-checkpoints] stage=trajectories steps=$steps"
        exec fish $launcher $forwarded
end
