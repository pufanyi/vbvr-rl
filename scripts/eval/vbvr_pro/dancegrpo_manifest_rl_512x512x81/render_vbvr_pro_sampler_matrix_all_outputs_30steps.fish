#!/usr/bin/env fish

# Render the full 500-sample denoising trajectory for every model x sampler
# cell in the native-512 VBVR-Pro matrix.  Each sample directory contains
# step_00..step_29, the exact formal final, a 30-cell grid video, a contact
# sheet, and a manifest.  The job is resumable sample-by-sample and deliberately
# leaves the files unpacked (no tar archive).

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -q CHECKPOINT_ROOT[1]
or set -gx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
set -q CONVERTED_BASE[1]
or set -g CONVERTED_BASE storage/models/dcp_converted_5b
set -q CONVERTED_PREFIX[1]
or set -g CONVERTED_PREFIX dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
set -q BASELINE_MODEL[1]
or set -g BASELINE_MODEL storage/models/diffsynth_converted_5b/wan2.2-TI2V-5B_260715_vbvr_pro_step-35500
set -q EVAL_OUTPUT_BASE[1]
or set -g EVAL_OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028
set -q TRAJECTORY_ROOT[1]
or set -g TRAJECTORY_ROOT storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories
set -q TRAJECTORY_LOG_DIR[1]
or set -g TRAJECTORY_LOG_DIR storage/eval_logs/vbvr_pro_sampler_matrix_all_500_30step_trajectories
if set -q TRAJECTORY_LOCAL_GPU_COUNT[1]
    string match -qr '^[1-9][0-9]*$' -- "$TRAJECTORY_LOCAL_GPU_COUNT"
    or _fail "TRAJECTORY_LOCAL_GPU_COUNT must be a positive integer, got '$TRAJECTORY_LOCAL_GPU_COUNT'"
    set -g TRAJECTORY_CUDA_DEVICES (seq 0 (math $TRAJECTORY_LOCAL_GPU_COUNT - 1))
else if set -q TRAJECTORY_CUDA_DEVICES[1]
    # Exported Fish lists cross an `exec fish` boundary as one scalar. Accept
    # either a native in-process list or a comma/whitespace-delimited scalar.
    if test (count $TRAJECTORY_CUDA_DEVICES) -eq 1
        set -g TRAJECTORY_CUDA_DEVICES (string match -ra '[^,[:space:]]+' -- "$TRAJECTORY_CUDA_DEVICES")
    end
else
    set -g TRAJECTORY_CUDA_DEVICES 0 1 2 3 4 5 6 7
end
set -q TRAJECTORY_WORKERS_PER_GPU[1]
or set -g TRAJECTORY_WORKERS_PER_GPU 1
string match -qr '^[1-9][0-9]*$' -- "$TRAJECTORY_WORKERS_PER_GPU"
or _fail "TRAJECTORY_WORKERS_PER_GPU must be a positive integer, got '$TRAJECTORY_WORKERS_PER_GPU'"
set -q TRAJECTORY_PROGRESS_INTERVAL[1]
or set -g TRAJECTORY_PROGRESS_INTERVAL 30
string match -qr '^[1-9][0-9]*$' -- "$TRAJECTORY_PROGRESS_INTERVAL"
or _fail "TRAJECTORY_PROGRESS_INTERVAL must be a positive integer, got '$TRAJECTORY_PROGRESS_INTERVAL'"
# This is the immutable checkpoint snapshot captured by the quantitative
# matrix launcher at 2026-08-06 18:26 +08:00. Later training checkpoints must
# first receive the same six formal evaluations before being added explicitly.
set -q MATRIX_CHECKPOINT_STEPS[1]
or set -g MATRIX_CHECKPOINT_STEPS 100 200 300 400 500 600 700 800 900
set -q MATRIX_INCLUDE_BASELINE[1]
or set -g MATRIX_INCLUDE_BASELINE 1
contains -- $MATRIX_INCLUDE_BASELINE 0 1
or _fail "MATRIX_INCLUDE_BASELINE must be 0 or 1: $MATRIX_INCLUDE_BASELINE"
set -q GT_BASE[1]
or set -g GT_BASE (realpath storage/datasets/vbvr-pro-eval-500)
set -q EVALKIT_REV[1]
or set -g EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -q EVALKIT_SOURCE_SHA256[1]
or set -g EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8

set -g _manifest_sha256 afab352e08c590c9f4b480ef314b37f6896eef6430f42ea6c0ce0494f2aa8c4e
set -g _trajectory_converted_base $CONVERTED_BASE
set -g _trajectory_converted_prefix $CONVERTED_PREFIX
set -g _trajectory_baseline_model $BASELINE_MODEL
set -g _trajectory_eval_base $EVAL_OUTPUT_BASE
set -g _trajectory_root $TRAJECTORY_ROOT
set -g _trajectory_log_dir $TRAJECTORY_LOG_DIR
set -g _trajectory_cuda_devices $TRAJECTORY_CUDA_DEVICES
set -g _trajectory_workers_per_gpu $TRAJECTORY_WORKERS_PER_GPU
set -g _trajectory_progress_interval $TRAJECTORY_PROGRESS_INTERVAL

test (count $_trajectory_cuda_devices) -gt 0; or _fail "TRAJECTORY_CUDA_DEVICES must select at least one GPU"
test -d $GT_BASE; or _fail "GT_BASE does not exist: $GT_BASE"

set -l checkpoint_steps
set -l requested_steps (string match -ra '[^,[:space:]]+' -- (string join ',' -- $MATRIX_CHECKPOINT_STEPS))
for step in $requested_steps
    string match -qr '^[1-9][0-9]*$' -- "$step"
    or _fail "MATRIX_CHECKPOINT_STEPS contains an invalid step: $step"
    contains -- $step $checkpoint_steps; and _fail "MATRIX_CHECKPOINT_STEPS contains duplicate step $step"
    set -a checkpoint_steps $step
end
for step in $checkpoint_steps
    test -f $CHECKPOINT_ROOT/checkpoint-$step/high/.metadata
    or _fail "matrix checkpoint is missing or incomplete: checkpoint-$step"
end
test (count $checkpoint_steps) -gt 0; or _fail "MATRIX_CHECKPOINT_STEPS must not be empty"
set -l model_ids $checkpoint_steps
if test "$MATRIX_INCLUDE_BASELINE" = 1
    set -p model_ids baseline
end
set -l sampler_ids cps0p1 cps0p3 cps0p7 cps0p9 euler unipc

function _sampler_parts
    switch $argv[1]
        case cps0p1
            printf '%s\n' cps 0.1 cps-noise-0.1 cps unused
        case cps0p3
            printf '%s\n' cps 0.3 cps-noise-0.3 cps unused
        case cps0p7
            printf '%s\n' cps 0.7 cps-noise-0.7 cps unused
        case cps0p9
            printf '%s\n' cps 0.9 cps-noise-0.9 cps unused
        case euler
            printf '%s\n' euler unused euler-ode-30steps-cfg1 ode euler
        case unipc
            printf '%s\n' unipc unused unipc-ode-30steps-cfg1 ode unipc
        case '*'
            return 1
    end
end

function _model_path
    if test "$argv[1]" = baseline
        echo $_trajectory_baseline_model
    else
        echo $_trajectory_converted_base/$_trajectory_converted_prefix"_checkpoint-$argv[1]"
    end
end

function _eval_root
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l label $parts[3]
    if test "$model_id" = baseline
        if test "$sampler_id" = cps0p7
            echo $_trajectory_eval_base/diffsynth_step35500-baseline-cps0p7-30steps-cfg1
        else
            echo $_trajectory_eval_base/diffsynth_step35500-baseline-$label
        end
    else
        echo $_trajectory_eval_base/dancegrpo_vbvr_pro_5b_checkpoint-$model_id-$label
    end
end

function _trajectory_dir
    echo $_trajectory_root/$argv[1]-$argv[2]
end

function _common_render_args
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l shard_index (set -q argv[3]; and echo $argv[3]; or echo 0)
    set -l shard_count (set -q argv[4]; and echo $argv[4]; or echo 1)
    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l mode $parts[1]
    set -l level $parts[2]
    set -l eval_root (_eval_root $model_id $sampler_id); or return 1
    set -l noise_args
    if test "$mode" = cps
        set noise_args --noise_level $level
    end
    set -l limit_args
    if set -q TRAJECTORY_LIMIT[1]
        set limit_args --limit $TRAJECTORY_LIMIT
    end
    printf '%s\n' \
        --eval_json $eval_root/eval_samples.json \
        --model_path (_model_path $model_id) \
        --output_dir (_trajectory_dir $model_id $sampler_id) \
        --sampler $mode \
        $noise_args \
        --all_samples \
        --sample_shard_index $shard_index \
        --sample_shard_count $shard_count \
        --formal_final_root $eval_root/generated_512x512x81 \
        $limit_args \
        --height 512 --width 512 --num_frames 81 \
        --num_inference_steps 30 --guidance_scale 1.0 \
        --fps 16 --seed 0 --device cuda:0 \
        --grid_cols 6 --grid_thumb_width 160
end

function _formal_complete
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l level $parts[2]
    set -l generation_mode $parts[4]
    set -l ode_solver $parts[5]
    set -l generation_args --generation-mode $generation_mode
    if test "$generation_mode" = cps
        set -a generation_args --cps-noise-level $level
    else
        set -a generation_args --ode-solver $ode_solver
    end
    .venv/bin/python -m src.cli.audit_vbvr_sampler_run \
        --output-root (_eval_root $model_id $sampler_id) \
        --converted-model (_model_path $model_id) \
        --gt-base $GT_BASE \
        --manifest-sha256 $_manifest_sha256 \
        --evalkit-revision $EVALKIT_REV \
        --evalkit-source-sha256 $EVALKIT_SOURCE_SHA256 \
        $generation_args \
        --fast --quiet
end

function _trajectory_complete
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l mode $parts[1]
    set -l level $parts[2]
    set -l eval_root (_eval_root $model_id $sampler_id); or return 1
    set -l noise_args
    if test "$mode" = cps
        set noise_args --noise_level $level
    end
    set -l limit_args
    if set -q TRAJECTORY_LIMIT[1]
        set limit_args --limit $TRAJECTORY_LIMIT
    end
    .venv/bin/python -m src.cli.audit_vbvr_i2v_trajectories \
        --eval_json $eval_root/eval_samples.json \
        --model_path (_model_path $model_id) \
        --output_dir (_trajectory_dir $model_id $sampler_id) \
        --formal_final_root $eval_root/generated_512x512x81 \
        --sampler $mode \
        $noise_args \
        $limit_args \
        --height 512 --width 512 --num_frames 81 \
        --num_inference_steps 30 --guidance_scale 1.0 \
        --fps 16 --seed 0 --quiet \
        >/dev/null 2>&1
end

function _finalize_trajectory_cell
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l mode $parts[1]
    set -l level $parts[2]
    set -l eval_root (_eval_root $model_id $sampler_id); or return 1
    set -l noise_args
    if test "$mode" = cps
        set noise_args --noise_level $level
    end
    set -l limit_args
    if set -q TRAJECTORY_LIMIT[1]
        set limit_args --limit $TRAJECTORY_LIMIT
    end
    .venv/bin/python -m src.cli.audit_vbvr_i2v_trajectories \
        --eval_json $eval_root/eval_samples.json \
        --model_path (_model_path $model_id) \
        --output_dir (_trajectory_dir $model_id $sampler_id) \
        --formal_final_root $eval_root/generated_512x512x81 \
        --sampler $mode \
        $noise_args \
        $limit_args \
        --height 512 --width 512 --num_frames 81 \
        --num_inference_steps 30 --guidance_scale 1.0 \
        --fps 16 --seed 0 --write-cell-manifest --quiet
end

set -l tasks
for sampler_id in $sampler_ids
    if set -q SAMPLER_FILTER[1]; and test "$SAMPLER_FILTER" != "$sampler_id"
        continue
    end
    for model_id in $model_ids
        if set -q MODEL_FILTER[1]; and test "$MODEL_FILTER" != "$model_id"
            continue
        end
        set -a tasks "$model_id,$sampler_id"
    end
end
test (count $tasks) -gt 0; or _fail "filters selected no trajectory tasks"

# A scheduler-facing wrapper can shard the immutable cell list by node. Keep
# this launcher usable on one workstation by default, and shard only after the
# model/sampler filters have been applied. Round-robin assignment balances the
# 60-cell matrix across nodes; with WORLD_SIZE=8 each node receives 7 or 8
# cells, so every selected local GPU runs at most one cell in the common case.
set -l trajectory_node_count 1
set -l trajectory_node_rank 0
if set -q TRAJECTORY_NODE_COUNT[1]
    string match -qr '^[1-9][0-9]*$' -- "$TRAJECTORY_NODE_COUNT"
    or _fail "TRAJECTORY_NODE_COUNT must be a positive integer, got '$TRAJECTORY_NODE_COUNT'"
    set trajectory_node_count $TRAJECTORY_NODE_COUNT
end
if set -q TRAJECTORY_NODE_RANK[1]
    string match -qr '^[0-9]+$' -- "$TRAJECTORY_NODE_RANK"
    or _fail "TRAJECTORY_NODE_RANK must be a non-negative integer, got '$TRAJECTORY_NODE_RANK'"
    set trajectory_node_rank $TRAJECTORY_NODE_RANK
end
test $trajectory_node_rank -lt $trajectory_node_count
or _fail "TRAJECTORY_NODE_RANK=$trajectory_node_rank is outside [0, $trajectory_node_count)"

set -l unsharded_task_count (count $tasks)
set -l node_tasks
for task_index in (seq $unsharded_task_count)
    if test (math "($task_index - 1) % $trajectory_node_count") -eq $trajectory_node_rank
        set -a node_tasks $tasks[$task_index]
    end
end
set tasks $node_tasks

set -l worker_tasks
for task_value in $tasks
    for shard_index in (seq 0 (math $_trajectory_workers_per_gpu - 1))
        set -a worker_tasks "$task_value,$shard_index,$_trajectory_workers_per_gpu"
    end
end

echo "[trajectory] node shard: rank=$trajectory_node_rank count=$trajectory_node_count"
echo "[trajectory] global selected cells before node sharding: $unsharded_task_count"
echo "[trajectory] node-assigned cells: "(count $tasks)
echo "[trajectory] workers per GPU/cell: $_trajectory_workers_per_gpu"
set -l assignment_wave_size (count $_trajectory_cuda_devices)
for task_index in (seq (count $worker_tasks))
    set -l assignment_slot (math "(($task_index - 1) % $assignment_wave_size) + 1")
    set -l worker (string split , -- $worker_tasks[$task_index])
    echo "[assignment] node=$trajectory_node_rank gpu=$_trajectory_cuda_devices[$assignment_slot] "\
        "task=$worker[1],$worker[2] sample_shard=$worker[3]/$worker[4]"
end

if test (count $tasks) -eq 0
    echo "[done] node $trajectory_node_rank has no assigned cells"
    exit 0
end
if set -q TRAJECTORY_ASSIGNMENT_ONLY[1]; and test "$TRAJECTORY_ASSIGNMENT_ONLY" = "1"
    echo "[done] assignment-only mode; no model was loaded and no output was written"
    exit 0
end

mkdir -p $_trajectory_root $_trajectory_log_dir; or _fail "could not create trajectory output/log directories"

echo "[preflight] checking that every selected quantitative cell passed its recorded-contract audit"
for task_value in $tasks
    set -l task (string split , -- $task_value)
    _formal_complete $task[1] $task[2]; or _fail "formal quantitative cell is incomplete: $task_value"
end

function _launch_wave
    set -l wave_tasks $argv
    set -l running_pids
    set -l running_tasks
    set -l running_logs
    set -l launched_cells
    set -l checked_cells
    set -l known_complete_cells
    for slot in (seq (count $wave_tasks))
        set -l task (string split , -- $wave_tasks[$slot])
        set -l model_id $task[1]
        set -l sampler_id $task[2]
        set -l shard_index $task[3]
        set -l shard_count $task[4]
        set -l cell "$model_id,$sampler_id"
        set -l log_path $_trajectory_log_dir/$model_id-$sampler_id-shard-(string pad -w 3 -c 0 $shard_index)-of-(string pad -w 3 -c 0 $shard_count).log
        set -l device_slot (math "(($slot - 1) % "(count $_trajectory_cuda_devices)") + 1")
        set -l device $_trajectory_cuda_devices[$device_slot]
        if contains -- $cell $known_complete_cells
            continue
        end
        if not contains -- $cell $checked_cells
            set -a checked_cells $cell
            if _trajectory_complete $model_id $sampler_id
                echo "[skip] $model_id/$sampler_id all selected trajectories already complete"
                set -a known_complete_cells $cell
                continue
            end
        end

        set -l args (_common_render_args $model_id $sampler_id $shard_index $shard_count); or return 1
        echo "[start] "(date --iso-8601=seconds)" $model_id/$sampler_id "\
            "shard=$shard_index/$shard_count GPU=$device log=$log_path"
        printf '\n[start] %s model=%s sampler=%s shard=%s/%s gpu=%s\n' \
            (date --iso-8601=seconds) $model_id $sampler_id $shard_index $shard_count $device >>$log_path
        env CUDA_VISIBLE_DEVICES=$device \
            .venv/bin/python -m src.cli.render_vbvr_i2v_steps $args \
            >>$log_path 2>&1 &
        set -a running_pids $last_pid
        set -a running_tasks $wave_tasks[$slot]
        set -a running_logs $log_path
        if not contains -- $cell $launched_cells
            set -a launched_cells $cell
        end
    end

    set -l progress_pid
    if test (count $running_pids) -gt 0
        set -l progress_args \
            --trajectory-root $_trajectory_root \
            --shard-count $_trajectory_workers_per_gpu \
            --samples-per-cell (set -q TRAJECTORY_LIMIT[1]; and echo $TRAJECTORY_LIMIT; or echo 500) \
            --interval $_trajectory_progress_interval
        for cell in $launched_cells
            set -a progress_args --cell $cell
        end
        .venv/bin/python -m src.cli.watch_vbvr_trajectory_progress $progress_args --watch &
        set progress_pid $last_pid
    end

    set -l failed 0
    set -l failed_cells
    for slot in (seq (count $running_pids))
        wait $running_pids[$slot]
        set -l rc $status
        set -l task (string split , -- $running_tasks[$slot])
        set -l cell "$task[1],$task[2]"
        if test $rc -ne 0
            set failed 1
            if not contains -- $cell $failed_cells
                set -a failed_cells $cell
            end
            echo "[error] $task[1]/$task[2] shard=$task[3]/$task[4] failed; log=$running_logs[$slot]" >&2
            tail -n 160 $running_logs[$slot] >&2
        end
    end
    if test -n "$progress_pid"
        kill $progress_pid >/dev/null 2>&1
        wait $progress_pid >/dev/null 2>&1
        # The watcher may be sleeping when the last worker finishes. Always
        # emit one final snapshot so terminal/log output reaches N/N.
        .venv/bin/python -m src.cli.watch_vbvr_trajectory_progress $progress_args --compact
    end

    for cell in $launched_cells
        set -l task (string split , -- $cell)
        if contains -- $cell $failed_cells
            continue
        end
        if not _finalize_trajectory_cell $task[1] $task[2]; or not _trajectory_complete $task[1] $task[2]
            set failed 1
            echo "[error] $task[1]/$task[2] failed the final strict cell audit" >&2
        else
            echo "[done]  "(date --iso-8601=seconds)" $task[1]/$task[2]"
        end
    end
    test $failed -eq 0
end

echo "[trajectory] models: $model_ids"
echo "[trajectory] samplers: $sampler_ids"
echo "[trajectory] selected cells: "(count $tasks)
echo "[trajectory] concurrent workers: "(count $worker_tasks)
echo "[trajectory] selected samples/cell: "(set -q TRAJECTORY_LIMIT[1]; and echo $TRAJECTORY_LIMIT; or echo 500)
echo "[trajectory] output root: $_trajectory_root"
echo "[trajectory] files remain unpacked; no tar archive will be created"

set -l wave_size (math (count $_trajectory_cuda_devices) \* $_trajectory_workers_per_gpu)
set -l start 1
while test $start -le (count $worker_tasks)
    set -l wave
    for offset in (seq 0 (math $wave_size - 1))
        set -l position (math $start + $offset)
        if test $position -le (count $worker_tasks)
            set -a wave $worker_tasks[$position]
        end
    end
    echo "[wave] $wave"
    _launch_wave $wave; or _fail "trajectory wave failed: $wave"
    set start (math $start + $wave_size)
end

echo "[done] every selected model output has a validated 30-step trajectory under $_trajectory_root"
