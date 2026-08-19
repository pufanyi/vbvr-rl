#!/usr/bin/env fish

# Thin experiment adapter for the native-512 Qwen3.6-VLM DanceGRPO run.
# Checkpoint discovery, strict six-sampler audits, conversion, formal evaluation,
# and optional all-output trajectories are delegated to the shared native-512
# incremental evaluator. After `formal`, all nodes wait for the same strict
# six-sampler snapshot and automatically score only missing cells with the
# training-time Qwen3.6 judge. This file selects the DCP source and isolates all
# derived model, score, log, trajectory, and VLM-judge namespaces by run name.
# When no scheduler topology is present, this adapter defaults to one machine;
# --nproc remains the number of GPUs available on that machine.

function _fail
    echo "[error] $argv" >&2
    exit 1
end

function _usage
    echo "Usage:"
    echo "  fish "(status filename)" [formal|trajectories] [options]"
    echo
    echo "Source selection (choose at most one):"
    echo "  --checkpoint-dir PATH   Evaluate exactly one complete checkpoint-N directory"
    echo "  --checkpoint-root PATH  Incrementally discover complete checkpoint-N children"
    echo
    echo "Shared evaluator options include:"
    echo "  --checkpoints N[,N...]  Restrict a checkpoint root to explicit steps"
    echo "  --nproc N               Local GPU count for formal evaluation (default: 8)"
    echo "  --assignment-only       Print the multi-node assignment without evaluation"
    echo
    echo "Evaluation topology:"
    echo "  Omit WORLD_SIZE and RANK on one machine (defaults: WORLD_SIZE=1, RANK=0)."
    echo "  On multiple machines, set both variables on every node."
    echo
    echo "Automatic VLM judge options:"
    echo "  --no-vlm-judge          Stop after formal EvalKit evaluation"
    echo "  --vlm-output-root PATH  Override the independent resumable judge result root"
    echo "  --vlm-concurrency N     Judge requests per node (default: 2 x --nproc)"
    echo
    echo "The formal stage waits for every node's strict cells, then judges only missing"
    echo "cells. If this node has no pending judge cells, it never starts Qwen."
    echo
    echo "VLM_EVAL_CHECKPOINT_ROOT remains a supported root override when neither source option is used."
end

set -l script_dir (realpath (dirname (status filename)))
set -l project_root (realpath $script_dir/../../../..)
cd $project_root; or _fail "could not enter project root: $project_root"

set -l has_world_size 0
set -l has_rank 0
set -q WORLD_SIZE[1]; and set has_world_size 1
set -q RANK[1]; and set has_rank 1
test $has_world_size -eq $has_rank
or _fail "WORLD_SIZE and RANK must be set together; omit both for single-node evaluation"
if test $has_world_size -eq 0
    set -gx WORLD_SIZE 1
    set -gx RANK 0
    echo "[topology] WORLD_SIZE/RANK not set; using single node 0/1"
end

# This is the unsuffixed YAML output. Scale-out training adds a topology suffix;
# point VLM_EVAL_CHECKPOINT_ROOT or --checkpoint-root at that run.
set -l default_checkpoint_root \
    storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_vlm_qwen36_task_prompts_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl

set -l checkpoint_dir
set -l checkpoint_root_arg
set -l checkpoint_dir_seen 0
set -l checkpoint_root_seen 0
set -l stage
set -l forwarded
set -l forwarded_checkpoint_filter 0
set -l checkpoint_filter
set -l assignment_only 0
set -l nproc 8
set -l auto_vlm_judge (set -q VLM_EVAL_AUTO_JUDGE; and echo $VLM_EVAL_AUTO_JUDGE; or echo 1)
set -l vlm_output_root (set -q VLM_JUDGE_OUTPUT_ROOT; and echo $VLM_JUDGE_OUTPUT_ROOT; or echo '')
set -l vlm_concurrency (set -q VLM_JUDGE_CONCURRENCY; and echo $VLM_JUDGE_CONCURRENCY; or echo '')
set -l expect_value
for arg in $argv
    if test -n "$expect_value"
        switch $expect_value
            case checkpoint_dir
                set checkpoint_dir $arg
            case checkpoint_root
                set checkpoint_root_arg $arg
            case checkpoints
                test $forwarded_checkpoint_filter -eq 0
                or _fail "--checkpoints may be specified only once"
                set forwarded_checkpoint_filter 1
                set checkpoint_filter $arg
                set -a forwarded --checkpoints $arg
            case nproc
                set nproc $arg
                set -a forwarded --nproc $arg
            case vlm_output_root
                set vlm_output_root $arg
            case vlm_concurrency
                set vlm_concurrency $arg
        end
        set expect_value
        continue
    end

    switch $arg
        case -h --help
            _usage
            exit 0
        case formal trajectories
            test -z "$stage"; or _fail "specify exactly one stage: formal or trajectories"
            set stage $arg
        case --checkpoint-dir
            test $checkpoint_dir_seen -eq 0; or _fail "--checkpoint-dir may be specified only once"
            set checkpoint_dir_seen 1
            set expect_value checkpoint_dir
        case '--checkpoint-dir=*'
            test $checkpoint_dir_seen -eq 0; or _fail "--checkpoint-dir may be specified only once"
            set checkpoint_dir_seen 1
            set checkpoint_dir (string replace -- '--checkpoint-dir=' '' "$arg")
        case --checkpoint-root
            test $checkpoint_root_seen -eq 0; or _fail "--checkpoint-root may be specified only once"
            set checkpoint_root_seen 1
            set expect_value checkpoint_root
        case '--checkpoint-root=*'
            test $checkpoint_root_seen -eq 0; or _fail "--checkpoint-root may be specified only once"
            set checkpoint_root_seen 1
            set checkpoint_root_arg (string replace -- '--checkpoint-root=' '' "$arg")
        case --checkpoints
            set expect_value checkpoints
        case '--checkpoints=*'
            test $forwarded_checkpoint_filter -eq 0
            or _fail "--checkpoints may be specified only once"
            set checkpoint_filter (string replace -- '--checkpoints=' '' "$arg")
            test -n "$checkpoint_filter"; or _fail "--checkpoints requires a value"
            set forwarded_checkpoint_filter 1
            set -a forwarded --checkpoints $checkpoint_filter
        case --nproc
            set expect_value nproc
        case '--nproc=*'
            set nproc (string replace -- '--nproc=' '' "$arg")
            test -n "$nproc"; or _fail "--nproc requires a value"
            set -a forwarded --nproc $nproc
        case --assignment-only
            set assignment_only 1
            set -a forwarded $arg
        case --no-vlm-judge
            set auto_vlm_judge 0
        case --vlm-output-root
            set expect_value vlm_output_root
        case '--vlm-output-root=*'
            set vlm_output_root (string replace -- '--vlm-output-root=' '' "$arg")
            test -n "$vlm_output_root"; or _fail "--vlm-output-root requires a value"
        case --vlm-concurrency
            set expect_value vlm_concurrency
        case '--vlm-concurrency=*'
            set vlm_concurrency (string replace -- '--vlm-concurrency=' '' "$arg")
            test -n "$vlm_concurrency"; or _fail "--vlm-concurrency requires a value"
        case '*'
            set -a forwarded $arg
    end
end
test -z "$expect_value"; or _fail "--"(string replace '_' '-' -- $expect_value)" requires a value"
test $checkpoint_dir_seen -eq 0 -o -n "$checkpoint_dir"
or _fail "--checkpoint-dir requires a value"
test $checkpoint_root_seen -eq 0 -o -n "$checkpoint_root_arg"
or _fail "--checkpoint-root requires a value"
test $checkpoint_dir_seen -eq 0 -o $checkpoint_root_seen -eq 0
or _fail "--checkpoint-dir and --checkpoint-root are mutually exclusive"
test -n "$stage"; or set stage formal
contains -- "$auto_vlm_judge" 0 1
or _fail "VLM_EVAL_AUTO_JUDGE must be 0 or 1: $auto_vlm_judge"
string match -qr '^[1-9][0-9]*$' -- "$nproc"
or _fail "--nproc must be a positive integer: $nproc"
if test "$stage" = formal
    test (math "$nproc % 2") -eq 0
    or _fail "--nproc must be even because every formal cell and the default Qwen topology use GPU pairs"
end
if test -z "$vlm_concurrency"
    set vlm_concurrency (math "$nproc * 2")
end
string match -qr '^[1-9][0-9]*$' -- "$vlm_concurrency"
or _fail "--vlm-concurrency must be a positive integer: $vlm_concurrency"

set -l checkpoint_root
set -l direct_step
if test $checkpoint_dir_seen -eq 1
    test $forwarded_checkpoint_filter -eq 0
    or _fail "--checkpoint-dir already selects one step; do not also pass --checkpoints"
    test -d "$checkpoint_dir"; or _fail "checkpoint directory does not exist: $checkpoint_dir"
    set checkpoint_dir (realpath "$checkpoint_dir")
    set -l checkpoint_name (basename "$checkpoint_dir")
    set direct_step (string match -r --groups-only '^checkpoint-([1-9][0-9]*)$' -- "$checkpoint_name")
    test -n "$direct_step"
    or _fail "--checkpoint-dir must end in checkpoint-<positive integer>: $checkpoint_dir"
    test -f "$checkpoint_dir/high/.metadata"
    or _fail "checkpoint is incomplete (missing high/.metadata): $checkpoint_dir"
    set checkpoint_root (dirname "$checkpoint_dir")
    set -a forwarded --checkpoints $direct_step
else if test $checkpoint_root_seen -eq 1
    set checkpoint_root $checkpoint_root_arg
else if set -q VLM_EVAL_CHECKPOINT_ROOT[1]
    set checkpoint_root $VLM_EVAL_CHECKPOINT_ROOT
else
    set checkpoint_root $default_checkpoint_root
end

if test -d "$checkpoint_root"
    set checkpoint_root (realpath "$checkpoint_root")
end
set -l source_name (basename (string trim -r -c / -- "$checkpoint_root"))
test -n "$source_name"; or _fail "could not derive a run name from checkpoint root: $checkpoint_root"

set -gx CHECKPOINT_ROOT $checkpoint_root
set -gx VLM_EVAL_CHECKPOINT_ROOT $checkpoint_root
set -q CONVERTED_BASE[1]
or set -gx CONVERTED_BASE storage/models/dcp_converted_5b
set -q CONVERTED_PREFIX[1]
or set -gx CONVERTED_PREFIX $source_name
set -q OUTPUT_BASE[1]
or set -gx OUTPUT_BASE \
    storage/eval_out/vbvr_pro_main_v2_512x512x81_$source_name"_eval500_manifest_afab352e_evalkit_4cc7d028"
set -q EVAL_LOG_DIR[1]
or set -gx EVAL_LOG_DIR storage/eval_logs/vbvr_pro_sampler_matrix_30steps_$source_name
set -q TRAJECTORY_ROOT[1]
or set -gx TRAJECTORY_ROOT \
    storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories_$source_name
set -q TRAJECTORY_LOG_DIR[1]
or set -gx TRAJECTORY_LOG_DIR \
    storage/eval_logs/vbvr_pro_sampler_matrix_all_500_30step_trajectories_$source_name

# Freeze checkpoint discovery once per process so its formal and judge scopes
# are identical. Scheduler nodes must start from the same launch snapshot; when
# training may publish a checkpoint during node startup, pass --checkpoints to
# make that cross-node snapshot explicit.
set -l invocation_steps
if test -n "$direct_step"
    set invocation_steps $direct_step
else if test $forwarded_checkpoint_filter -eq 1
    set invocation_steps (string match -ra '[^,[:space:]]+' -- "$checkpoint_filter")
else if test -d "$checkpoint_root"
    for candidate in (find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
        set -l candidate_name (basename "$candidate")
        set -l candidate_step (string match -r --groups-only '^checkpoint-([1-9][0-9]*)$' -- "$candidate_name")
        test -n "$candidate_step"; or continue
        test -f "$candidate/high/.metadata"; or continue
        set -a invocation_steps $candidate_step
    end
    if test (count $invocation_steps) -gt 0
        set checkpoint_filter (string join ',' -- $invocation_steps)
        set -a forwarded --checkpoints $checkpoint_filter
    end
end

set -l delegate \
    (set -q VLM_EVAL_SHARED_INCREMENTAL_LAUNCHER; and echo $VLM_EVAL_SHARED_INCREMENTAL_LAUNCHER; or echo \
        $script_dir/../dancegrpo_manifest_rl_512x512x81/evaluate_incremental_multinode.fish)
test -f "$delegate"; or _fail "shared incremental evaluator is missing: $delegate"

set -l selection "all complete numeric checkpoints"
if test -n "$direct_step"
    set selection checkpoint-$direct_step
else if test $forwarded_checkpoint_filter -eq 1
    set selection "explicit --checkpoints filter"
else if test (count $invocation_steps) -gt 0
    set selection "frozen invocation snapshot "(string join ',' -- $invocation_steps)
end
echo "[vlm-eval] source=$checkpoint_root selection=$selection"
echo "[vlm-eval] namespace=$source_name output=$OUTPUT_BASE"

# Trajectories retain the old thin-delegation behavior and never invoke Qwen.
if test "$stage" = trajectories
    exec fish "$delegate" trajectories $forwarded
end

fish "$delegate" formal $forwarded
set -l formal_status $status
if test $formal_status -ne 0
    echo "[error] formal evaluator exited with status $formal_status on node $RANK" >&2
    exit $formal_status
end
if test $assignment_only -eq 1
    echo "[vlm-judge] assignment-only mode; automatic judge was not launched"
    exit 0
end
if test "$auto_vlm_judge" = 0
    echo "[vlm-judge] automatic judge disabled; formal EvalKit stage is complete on this node"
    exit 0
end
if test (count $invocation_steps) -eq 0
    echo "[vlm-judge] no complete checkpoint was selected at invocation; nothing to judge"
    exit 0
end

# The shared formal launcher intentionally has no distributed process group.
# Before every node derives the judge shard, repeatedly run its strict,
# side-effect-free audit until all selected six-sampler cells are complete.
# This makes every node discover the same exact cell list and keeps Qwen from
# contending with Wan inference still running on a slower node.
set -l formal_wait_timeout \
    (set -q VLM_EVAL_FORMAL_WAIT_TIMEOUT_SECONDS; and echo $VLM_EVAL_FORMAL_WAIT_TIMEOUT_SECONDS; or echo 172800)
set -l formal_wait_poll \
    (set -q VLM_EVAL_FORMAL_WAIT_POLL_SECONDS; and echo $VLM_EVAL_FORMAL_WAIT_POLL_SECONDS; or echo 30)
string match -qr '^[1-9][0-9]*$' -- "$formal_wait_timeout"
or _fail "VLM_EVAL_FORMAL_WAIT_TIMEOUT_SECONDS must be a positive integer"
string match -qr '^[1-9][0-9]*$' -- "$formal_wait_poll"
or _fail "VLM_EVAL_FORMAL_WAIT_POLL_SECONDS must be a positive integer"

if test "$WORLD_SIZE" -gt 1
    set -l wait_started (date +%s)
    set -l previous_pending
    while true
        set -l audit_output (fish "$delegate" formal --assignment-only $forwarded 2>&1)
        set -l audit_status $status
        if test $audit_status -ne 0
            printf '%s\n' $audit_output >&2
            _fail "strict formal barrier audit failed on node $RANK"
        end
        set -l pending_line (string match -r '^\[discover\] formal new or incomplete: .*' -- $audit_output)
        test (count $pending_line) -eq 1
        or begin
            printf '%s\n' $audit_output >&2
            _fail "could not read strict formal completion from the shared evaluator"
        end
        if test "$pending_line" = "[discover] formal new or incomplete: none"
            echo "[vlm-judge] strict formal barrier complete on node $RANK/$WORLD_SIZE"
            break
        end
        if test "$pending_line" != "$previous_pending"
            echo "[vlm-judge] waiting for all formal nodes: $pending_line"
            set previous_pending $pending_line
        end
        set -l elapsed (math (date +%s) - $wait_started)
        test $elapsed -lt $formal_wait_timeout
        or _fail "timed out after $elapsed seconds waiting for strict formal completion: $pending_line"
        sleep $formal_wait_poll
    end
end

set -l judge_launcher \
    (set -q VLM_EVAL_JUDGE_LAUNCHER; and echo $VLM_EVAL_JUDGE_LAUNCHER; or echo \
        $script_dir/evaluate_vlm_judge_multinode.fish)
test -f "$judge_launcher"; or _fail "VLM judge launcher is missing: $judge_launcher"
set -l judge_args score --input-root $OUTPUT_BASE --concurrency $vlm_concurrency
if test -n "$vlm_output_root"
    set -a judge_args --output-root $vlm_output_root
end
set -l judge_labels \
    cps-noise-0.1 cps-noise-0.3 cps-noise-0.7 cps-noise-0.9 \
    euler-ode-30steps-cfg1 unipc-ode-30steps-cfg1
for step in $invocation_steps
    for label in $judge_labels
        set -a judge_args --cell dancegrpo_vbvr_pro_5b_checkpoint-$step-$label
    end
end

# Use the same local GPU count as formal evaluation. The validated production
# default remains TP2 x DP4 for --nproc 8; smaller even nodes get one DP replica
# per GPU pair unless the operator explicitly supplies a service topology.
set -q WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE
or set -gx WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE 2
string match -qr '^[1-9][0-9]*$' -- "$WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE"
or _fail "WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE must be a positive integer"
test (math "$nproc % $WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE") -eq 0
or _fail "WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE=$WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE must divide --nproc=$nproc"
set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE
or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE (math "$nproc / $WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE")
set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL
or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL $WAN_TRAINER_VLM_DATA_PARALLEL_SIZE
string match -qr '^[1-9][0-9]*$' -- "$WAN_TRAINER_VLM_DATA_PARALLEL_SIZE"
or _fail "WAN_TRAINER_VLM_DATA_PARALLEL_SIZE must be a positive integer"
set -l vlm_gpu_count (math "$WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE * $WAN_TRAINER_VLM_DATA_PARALLEL_SIZE")
test $vlm_gpu_count -le $nproc
or _fail "Qwen TP x DP requests $vlm_gpu_count GPUs, exceeding --nproc=$nproc"

echo "[vlm-judge] auditing/resuming selected cells: steps="(string join ',' -- $invocation_steps) \
    " node=$RANK/$WORLD_SIZE concurrency=$vlm_concurrency"
fish "$judge_launcher" $judge_args
set -l judge_status $status
if test $judge_status -ne 0
    echo "[error] automatic VLM judge exited with status $judge_status on node $RANK" >&2
end
exit $judge_status
