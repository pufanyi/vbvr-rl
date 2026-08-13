#!/usr/bin/env fish

# Reusable incremental multi-node evaluation for the native-512 rule-reward
# DanceGRPO run. `formal` discovers complete numeric DCP checkpoints and
# submits only checkpoints lacking at least one strict six-sampler result.
# `trajectories` selects checkpoints whose six formal cells are complete; the
# delegated renderer then strictly skips complete cells and resumes samples.
#
# Repeated invocations are resumable but must not overlap. Run this command on
# every evaluation node with the same WORLD_SIZE and RANK=0..WORLD_SIZE-1.
# Pass `--checkpoints 2200` (or a comma-separated list) to pin both stages to
# an explicit snapshot instead of evaluating every discovered checkpoint.

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -l stage
set -l forwarded
set -l assignment_only 0
set -l checkpoint_filter
set -l expect_value
for arg in $argv
    if test -n "$expect_value"
        switch $expect_value
            case checkpoints
                set checkpoint_filter $arg
        end
        set expect_value
        continue
    end
    switch $arg
        case formal trajectories
            test -z "$stage"; or _fail "specify exactly one stage: formal or trajectories"
            set stage $arg
        case --assignment-only
            set assignment_only 1
        case --checkpoints
            test -z "$checkpoint_filter"; or _fail "--checkpoints may be specified only once"
            set expect_value checkpoints
        case '*'
            set -a forwarded $arg
    end
end
test -z "$expect_value"; or _fail "--$expect_value requires a value"
test -n "$stage"; or set stage formal

set -l script_dir (realpath (dirname (status filename)))
set -l project_root (realpath $script_dir/../../../..)
cd $project_root; or _fail "could not enter project root: $project_root"

set -q WORLD_SIZE[1]; or _fail "WORLD_SIZE is not set"
set -q RANK[1]; or _fail "RANK is not set"
string match -qr '^[1-9][0-9]*$' -- "$WORLD_SIZE"
or _fail "WORLD_SIZE must be a positive integer: $WORLD_SIZE"
string match -qr '^[0-9]+$' -- "$RANK"
or _fail "RANK must be a non-negative integer: $RANK"
test $RANK -lt $WORLD_SIZE
or _fail "RANK=$RANK is outside [0, $WORLD_SIZE)"

set -q CHECKPOINT_ROOT[1]
or set -gx CHECKPOINT_ROOT \
    storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
set -q CONVERTED_BASE[1]
or set -gx CONVERTED_BASE storage/models/dcp_converted_5b
set -q CONVERTED_PREFIX[1]
or set -gx CONVERTED_PREFIX \
    dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
set -q OUTPUT_BASE[1]
or set -gx OUTPUT_BASE \
    storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028
set -q EVAL_LOG_DIR[1]
or set -gx EVAL_LOG_DIR storage/eval_logs/vbvr_pro_sampler_matrix_30steps_native512_e140
set -q TRAJECTORY_ROOT[1]
or set -gx TRAJECTORY_ROOT storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories
set -q TRAJECTORY_LOG_DIR[1]
or set -gx TRAJECTORY_LOG_DIR storage/eval_logs/vbvr_pro_sampler_matrix_all_500_30step_trajectories
set -q GT_BASE[1]
or set -gx GT_BASE (realpath storage/datasets/vbvr-pro-eval-500)
set -q EVALKIT_REV[1]
or set -gx EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -q EVALKIT_SOURCE_SHA256[1]
or set -gx EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
set -g _incremental_manifest_sha256 afab352e08c590c9f4b480ef314b37f6896eef6430f42ea6c0ce0494f2aa8c4e

if not test -d $CHECKPOINT_ROOT
    echo "[discover] checkpoint root does not exist yet: $CHECKPOINT_ROOT"
    echo "[done] no complete checkpoints are available; nothing to evaluate"
    exit 0
end

set -l complete_steps
for checkpoint_dir in (find $CHECKPOINT_ROOT -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
    set -l checkpoint_name (basename $checkpoint_dir)
    set -l step (string match -r --groups-only '^checkpoint-([1-9][0-9]*)$' -- $checkpoint_name)
    if test -z "$step"
        echo "[discover] ignoring non-numeric checkpoint alias: $checkpoint_name"
        continue
    end
    if not test -f $checkpoint_dir/high/.metadata
        echo "[discover] ignoring incomplete checkpoint without high/.metadata: $checkpoint_name"
        continue
    end
    set -a complete_steps $step
end
if test (count $complete_steps) -eq 0
    echo "[done] no complete numeric checkpoints under $CHECKPOINT_ROOT; nothing to evaluate"
    exit 0
end

if test -n "$checkpoint_filter"
    set -l requested_steps (string match -ra '[^,[:space:]]+' -- "$checkpoint_filter")
    test (count $requested_steps) -gt 0; or _fail "--checkpoints must contain at least one positive integer"
    set -l filtered_steps
    for step in $requested_steps
        string match -qr '^[1-9][0-9]*$' -- "$step"
        or _fail "--checkpoints contains an invalid step: $step"
        contains -- $step $filtered_steps; and _fail "--checkpoints contains duplicate step $step"
        contains -- $step $complete_steps
        or _fail "requested checkpoint-$step is missing or incomplete under $CHECKPOINT_ROOT"
        set -a filtered_steps $step
    end
    set complete_steps $filtered_steps
    echo "[discover] explicit checkpoint filter: "(string join ',' -- $complete_steps)
end

function _sampler_parts
    switch $argv[1]
        case cps0p1
            printf '%s\n' cps 0.1 unused cps-noise-0.1
        case cps0p3
            printf '%s\n' cps 0.3 unused cps-noise-0.3
        case cps0p7
            printf '%s\n' cps 0.7 unused cps-noise-0.7
        case cps0p9
            printf '%s\n' cps 0.9 unused cps-noise-0.9
        case euler
            printf '%s\n' ode unused euler euler-ode-30steps-cfg1
        case unipc
            printf '%s\n' ode unused unipc unipc-ode-30steps-cfg1
        case '*'
            return 1
    end
end

function _formal_cell_complete
    set -l step $argv[1]
    set -l sampler_id $argv[2]
    set -l converted $CONVERTED_BASE/$CONVERTED_PREFIX"_checkpoint-$step"
    test -f $converted/model_index.json; or return 1

    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l mode $parts[1]
    set -l level $parts[2]
    set -l solver $parts[3]
    set -l label $parts[4]
    set -l sampler_args --generation-mode $mode
    if test "$mode" = cps
        set -a sampler_args --cps-noise-level $level
    else
        set -a sampler_args --ode-solver $solver
    end

    .venv/bin/python -m src.cli.audit_vbvr_sampler_run \
        --output-root $OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$step-$label \
        --converted-model $converted \
        --gt-base $GT_BASE \
        --manifest-sha256 $_incremental_manifest_sha256 \
        --evalkit-revision $EVALKIT_REV \
        --evalkit-source-sha256 $EVALKIT_SOURCE_SHA256 \
        $sampler_args \
        --fast --quiet \
        >/dev/null 2>&1
end

set -l sampler_ids cps0p1 cps0p3 cps0p7 cps0p9 euler unipc
set -l formally_complete_steps
set -l formally_pending_steps
for step in $complete_steps
    set -l all_cells_complete 1
    for sampler_id in $sampler_ids
        if not _formal_cell_complete $step $sampler_id
            set all_cells_complete 0
            break
        end
    end
    if test $all_cells_complete -eq 1
        set -a formally_complete_steps $step
    else
        set -a formally_pending_steps $step
    end
end

set -l complete_text (string join ',' -- $complete_steps)
set -l formal_complete_text none
set -l formal_pending_text none
test (count $formally_complete_steps) -eq 0; or set formal_complete_text (string join ',' -- $formally_complete_steps)
test (count $formally_pending_steps) -eq 0; or set formal_pending_text (string join ',' -- $formally_pending_steps)
echo "[discover] source: $CHECKPOINT_ROOT"
echo "[discover] complete numeric checkpoints: $complete_text"
echo "[discover] strict six-sampler formal complete: $formal_complete_text"
echo "[discover] formal new or incomplete: $formal_pending_text"

set -l selected_steps
switch $stage
    case formal
        set selected_steps $formally_pending_steps
        if test (count $selected_steps) -eq 0
            echo "[done] every discovered checkpoint already has a strict complete six-sampler formal result"
            exit 0
        end
    case trajectories
        set selected_steps $formally_complete_steps
        if test (count $selected_steps) -eq 0
            echo "[done] no checkpoint has all six formal cells complete; no trajectory work is eligible"
            exit 0
        end
        if test (count $formally_pending_steps) -gt 0
            echo "[defer] trajectories require formal completion first: $formal_pending_text"
        end
end

# Export one comma-separated snapshot so every delegated child sees the same
# checkpoint list. Formal discovery selects only incomplete checkpoints. The
# trajectory launcher receives all formally eligible checkpoints and performs
# its own strict cell/sample audit, so completed cells never load a model.
set -gx MATRIX_CHECKPOINT_STEPS (string join ',' -- $selected_steps)
set -gx MATRIX_INCLUDE_BASELINE 0
if test $assignment_only -eq 0
    set -e MATRIX_ASSIGNMENT_ONLY
    set -e TRAJECTORY_ASSIGNMENT_ONLY
end

switch $stage
    case formal
        set -l launcher $script_dir/vbvr_pro_5b_sampler_matrix_30steps_multinode.fish
        set -l stage_args --checkpoints $MATRIX_CHECKPOINT_STEPS --no-baseline $forwarded
        if test $assignment_only -eq 1
            set -a stage_args --assignment-only
        end
        echo "[incremental] stage=formal node=$RANK/$WORLD_SIZE checkpoints=$MATRIX_CHECKPOINT_STEPS"
        exec fish $launcher $stage_args
    case trajectories
        set -gx EVAL_OUTPUT_BASE $OUTPUT_BASE
        set -l launcher $script_dir/render_vbvr_pro_sampler_matrix_all_outputs_30steps_multinode.fish
        if test $assignment_only -eq 1
            set -gx TRAJECTORY_ASSIGNMENT_ONLY 1
        end
        echo "[incremental] stage=trajectories node=$RANK/$WORLD_SIZE eligible_checkpoints=$MATRIX_CHECKPOINT_STEPS"
        exec fish $launcher $forwarded
end
