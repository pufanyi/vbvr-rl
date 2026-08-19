#!/usr/bin/env fish

# Matched native-512 VBVR-Pro sampler matrix:
#   baseline + every complete production checkpoint
#   Flow-CPS {0.1, 0.3, 0.7, 0.9}, Euler ODE, UniPC ODE
# All runs use 30 steps, CFG 1, seed 0, 512x512x81 at exact 16 FPS and the
# pinned e140 main_v2 scorer. Four two-GPU jobs fill one eight-H800 wave.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -l script_dir (dirname (status filename))
set -g _matrix_checkpoint_launcher $script_dir/../dancegrpo_manifest_rl_384x384x81/vbvr_pro_5b_dancegrpo_manifest_rl_checkpoint_cps0p7_main_v2.fish
set -g _matrix_baseline_launcher $script_dir/vbvr_pro_5b_diffsynth_step35500_baseline_cps0p7_main_v2.fish
test -f $_matrix_checkpoint_launcher; or _fail "checkpoint launcher missing: $_matrix_checkpoint_launcher"
test -f $_matrix_baseline_launcher; or _fail "baseline launcher missing: $_matrix_baseline_launcher"

set -q CHECKPOINT_ROOT[1]
or set -gx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_evalkit_e140038f
set -q CONVERTED_BASE[1]
or set -g CONVERTED_BASE storage/models/dcp_converted_5b
set -q CONVERTED_PREFIX[1]
or set -g CONVERTED_PREFIX dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_evalkit_e140038f
set -q BASELINE_MODEL[1]
or set -g BASELINE_MODEL storage/models/diffsynth_converted_5b/wan2.2-TI2V-5B_260715_vbvr_pro_step-35500
set -q OUTPUT_BASE[1]
or set -gx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028
set -q EVAL_LOG_DIR[1]
or set -g EVAL_LOG_DIR storage/eval_logs/vbvr_pro_sampler_matrix_30steps_native512_e140
set -q GT_BASE[1]
or set -gx GT_BASE (realpath storage/datasets/vbvr-pro-eval-500)
set -q SPLIT_MANIFEST[1]
or set -gx SPLIT_MANIFEST $GT_BASE/split_manifest.json
set -q EVALKIT_REV[1]
or set -gx EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -q EVALKIT_SOURCE_SHA256[1]
or set -gx EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
set -q EASYOCR_ROOT[1]
or set -gx EASYOCR_ROOT storage/evalkits/easyocr-shared
set -q EASYOCR_SOURCE_MODELS[1]
or set -gx EASYOCR_SOURCE_MODELS $EASYOCR_ROOT/model
set -q MATRIX_INCLUDE_BASELINE[1]
or set -g MATRIX_INCLUDE_BASELINE 1
contains -- $MATRIX_INCLUDE_BASELINE 0 1
or _fail "MATRIX_INCLUDE_BASELINE must be 0 or 1: $MATRIX_INCLUDE_BASELINE"
set -q MATRIX_LOCAL_GPU_COUNT[1]
or set -g MATRIX_LOCAL_GPU_COUNT 8
string match -qr '^[1-9][0-9]*$' -- "$MATRIX_LOCAL_GPU_COUNT"
or _fail "MATRIX_LOCAL_GPU_COUNT must be a positive integer: $MATRIX_LOCAL_GPU_COUNT"
test (math "$MATRIX_LOCAL_GPU_COUNT % 2") -eq 0
or _fail "MATRIX_LOCAL_GPU_COUNT must be even because each formal evaluation uses two GPUs"

set -l matrix_node_count 1
set -l matrix_node_rank 0
if set -q MATRIX_NODE_COUNT[1]
    string match -qr '^[1-9][0-9]*$' -- "$MATRIX_NODE_COUNT"
    or _fail "MATRIX_NODE_COUNT must be a positive integer: $MATRIX_NODE_COUNT"
    set matrix_node_count $MATRIX_NODE_COUNT
end
if set -q MATRIX_NODE_RANK[1]
    string match -qr '^[0-9]+$' -- "$MATRIX_NODE_RANK"
    or _fail "MATRIX_NODE_RANK must be a non-negative integer: $MATRIX_NODE_RANK"
    set matrix_node_rank $MATRIX_NODE_RANK
end
test $matrix_node_rank -lt $matrix_node_count
or _fail "MATRIX_NODE_RANK=$matrix_node_rank is outside [0, $matrix_node_count)"

set -g _matrix_device_pairs
for first_device in (seq 0 2 (math $MATRIX_LOCAL_GPU_COUNT - 2))
    set -a _matrix_device_pairs "$first_device,"(math $first_device + 1)
end

set -g _matrix_manifest_sha256 afab352e08c590c9f4b480ef314b37f6896eef6430f42ea6c0ce0494f2aa8c4e
set -g _matrix_output_base $OUTPUT_BASE
set -g _matrix_log_dir $EVAL_LOG_DIR
set -g _matrix_converted_base $CONVERTED_BASE
set -g _matrix_converted_prefix $CONVERTED_PREFIX
set -g _matrix_baseline_model $BASELINE_MODEL

test -d $CHECKPOINT_ROOT; or _fail "checkpoint root does not exist: $CHECKPOINT_ROOT"
test -d $GT_BASE; or _fail "GT_BASE does not exist: $GT_BASE"
test -f $SPLIT_MANIFEST; or _fail "split manifest does not exist: $SPLIT_MANIFEST"
test -d $BASELINE_MODEL; or _fail "baseline model does not exist: $BASELINE_MODEL"
mkdir -p $_matrix_log_dir; or _fail "could not create log directory: $_matrix_log_dir"

set -l actual_manifest_sha256 (sha256sum $SPLIT_MANIFEST | awk '{print $1}')
test "$actual_manifest_sha256" = "$_matrix_manifest_sha256"
or _fail "unexpected sanitized manifest SHA-256: $actual_manifest_sha256"
set -l checksums_sha256 (sha256sum $GT_BASE/SHA256SUMS | awk '{print $1}')
test "$checksums_sha256" = a67c534293724ddfc6657af755ab65e9b1354879deb2cfc47de22ede43942861
or _fail "unexpected dataset checksum-manifest SHA-256: $checksums_sha256"

if not set -q MATRIX_ASSIGNMENT_ONLY[1]
    echo "[dataset] verifying the complete downloaded VBVR-Pro eval snapshot"
    pushd $GT_BASE >/dev/null; or exit 1
    sha256sum -c SHA256SUMS --quiet
    set -l checksum_status $status
    popd >/dev/null; or exit 1
    test $checksum_status -eq 0; or _fail "VBVR-Pro eval snapshot failed SHA-256 verification"
    set -gx WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED 1
end

set -l checkpoint_steps
if set -q MATRIX_CHECKPOINT_STEPS[1]
    set -l requested_steps (string match -ra '[^,[:space:]]+' -- (string join ',' -- $MATRIX_CHECKPOINT_STEPS))
    for step in $requested_steps
        string match -qr '^[1-9][0-9]*$' -- "$step"
        or _fail "MATRIX_CHECKPOINT_STEPS contains an invalid step: $step"
        contains -- $step $checkpoint_steps; and _fail "MATRIX_CHECKPOINT_STEPS contains duplicate step $step"
        test -f $CHECKPOINT_ROOT/checkpoint-$step/high/.metadata
        or _fail "requested checkpoint is missing or incomplete: checkpoint-$step"
        set -a checkpoint_steps $step
    end
else
    for checkpoint_dir in (find $CHECKPOINT_ROOT -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
        test -f $checkpoint_dir/high/.metadata; or begin
            echo "[skip] incomplete checkpoint: $checkpoint_dir" >&2
            continue
        end
        set -a checkpoint_steps (string replace 'checkpoint-' '' -- (basename $checkpoint_dir))
    end
end
test (count $checkpoint_steps) -gt 0; or _fail "no complete checkpoints under $CHECKPOINT_ROOT"

function _converted_model
    set -l model_id $argv[1]
    if test "$model_id" = baseline
        echo $_matrix_baseline_model
    else
        echo $_matrix_converted_base/$_matrix_converted_prefix"_checkpoint-$model_id"
    end
end

function _sampler_parts
    switch $argv[1]
        case cps0p1
            printf '%s\n' cps 0.1 unipc cps-noise-0.1
        case cps0p3
            printf '%s\n' cps 0.3 unipc cps-noise-0.3
        case cps0p7
            printf '%s\n' cps 0.7 unipc cps-noise-0.7
        case cps0p9
            printf '%s\n' cps 0.9 unipc cps-noise-0.9
        case euler
            printf '%s\n' ode 0.7 euler euler-ode-30steps-cfg1
        case unipc
            printf '%s\n' ode 0.7 unipc unipc-ode-30steps-cfg1
        case '*'
            return 1
    end
end

function _output_root
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l label $parts[4]
    if test "$model_id" = baseline
        if test "$sampler_id" = cps0p7
            echo $_matrix_output_base/diffsynth_step35500-baseline-cps0p7-30steps-cfg1
        else
            echo $_matrix_output_base/diffsynth_step35500-baseline-$label
        end
    else
        echo $_matrix_output_base/dancegrpo_vbvr_pro_5b_checkpoint-$model_id-$label
    end
end

function _task_complete
    if set -q DRY_RUN[1]
        return 0
    end
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l mode $parts[1]
    set -l level $parts[2]
    set -l solver $parts[3]
    set -l sampler_args --generation-mode $mode
    if test "$mode" = cps
        set -a sampler_args --cps-noise-level $level
    else
        set -a sampler_args --ode-solver $solver
    end
    # This command is a boolean resume predicate. Missing artifacts are the
    # normal first-run case, so keep the auditor's exception traceback out of
    # operator logs; a launched task still reports real failures from its log.
    .venv/bin/python -m src.cli.audit_vbvr_sampler_run \
        --output-root (_output_root $model_id $sampler_id) \
        --converted-model (_converted_model $model_id) \
        --gt-base $GT_BASE \
        --manifest-sha256 $_matrix_manifest_sha256 \
        --evalkit-revision $EVALKIT_REV \
        --evalkit-source-sha256 $EVALKIT_SOURCE_SHA256 \
        $sampler_args \
        --fast --quiet \
        >/dev/null 2>&1
end

# Convert any newly completed checkpoint once before multiple sampler jobs try
# to share it. Existing models/provenance are validated and skipped by the
# regular conversion stage.
set -l conversion_index 0
for step in $checkpoint_steps
    set conversion_index (math $conversion_index + 1)
    if test (math "($conversion_index - 1) % $matrix_node_count") -ne $matrix_node_rank
        continue
    end
    set -l converted (_converted_model $step)
    if not set -q MATRIX_ASSIGNMENT_ONLY[1]; and not set -q DRY_RUN[1]; and not test -f $converted/model_index.json
        echo "[convert-preflight] checkpoint-$step -> $converted"
        env \
            CHECKPOINT_STEP=$step \
            CHECKPOINT_ROOT=$CHECKPOINT_ROOT \
            CONVERTED_MODEL=$converted \
            GT_BASE=$GT_BASE \
            SPLIT_MANIFEST=$SPLIT_MANIFEST \
            EVALKIT_REV=$EVALKIT_REV \
            EVALKIT_SOURCE_SHA256=$EVALKIT_SOURCE_SHA256 \
            EASYOCR_ROOT=$EASYOCR_ROOT \
            EASYOCR_SOURCE_MODELS=$EASYOCR_SOURCE_MODELS \
            WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED=1 \
            NUM_GPUS=1 \
            CUDA_DEVICES=0 \
            CONVERSION_ONLY=1 \
            fish $_matrix_checkpoint_launcher
        or _fail "conversion preflight failed for checkpoint-$step"
    end
end

set -l model_ids $checkpoint_steps
if test "$MATRIX_INCLUDE_BASELINE" = 1
    set -p model_ids baseline
end
set -l sampler_ids cps0p1 cps0p3 cps0p7 cps0p9 euler unipc
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
test (count $tasks) -gt 0; or _fail "filters selected no matrix tasks"

set -l global_task_count (count $tasks)
set -l node_tasks
for task_index in (seq $global_task_count)
    if test (math "($task_index - 1) % $matrix_node_count") -eq $matrix_node_rank
        set -a node_tasks $tasks[$task_index]
    end
end
set tasks $node_tasks

echo "[matrix] node shard: rank=$matrix_node_rank count=$matrix_node_count"
echo "[matrix] global selected tasks: $global_task_count"
echo "[matrix] node-assigned tasks: "(count $tasks)
for task_index in (seq (count $tasks))
    set -l device_slot (math "(($task_index - 1) % "(count $_matrix_device_pairs)") + 1")
    echo "[assignment] node=$matrix_node_rank GPUs=$_matrix_device_pairs[$device_slot] task=$tasks[$task_index]"
end
if test (count $tasks) -eq 0
    echo "[done] node $matrix_node_rank has no assigned formal matrix tasks"
    exit 0
end
if set -q MATRIX_ASSIGNMENT_ONLY[1]; and test "$MATRIX_ASSIGNMENT_ONLY" = 1
    echo "[done] assignment-only mode; no model was converted or evaluated"
    exit 0
end

function _launch_wave
    set -l wave_tasks $argv
    set -l running_pids
    set -l running_tasks
    set -l running_logs

    for slot in (seq (count $wave_tasks))
        set -l task_parts (string split , -- $wave_tasks[$slot])
        set -l model_id $task_parts[1]
        set -l sampler_id $task_parts[2]
        set -l parts (_sampler_parts $sampler_id); or return 1
        set -l mode $parts[1]
        set -l level $parts[2]
        set -l solver $parts[3]
        set -l devices $_matrix_device_pairs[$slot]
        set -l output_root (_output_root $model_id $sampler_id)
        set -l converted (_converted_model $model_id)
        set -l log_path $_matrix_log_dir/$model_id-$sampler_id.log

        if not set -q DRY_RUN[1]; and _task_complete $model_id $sampler_id
            echo "[skip] $model_id/$sampler_id already complete: $output_root"
            continue
        end

        set -l launcher $_matrix_checkpoint_launcher
        set -l model_args
        if test "$model_id" = baseline
            set launcher $_matrix_baseline_launcher
        else
            set model_args CHECKPOINT_STEP=$model_id CHECKPOINT_ROOT=$CHECKPOINT_ROOT
        end

        echo "[start] "(date --iso-8601=seconds)" $model_id/$sampler_id GPUs=$devices log=$log_path"
        env \
            $model_args \
            CONVERTED_MODEL=$converted \
            OUTPUT_BASE=$_matrix_output_base \
            OUTPUT_ROOT=$output_root \
            GT_BASE=$GT_BASE \
            SPLIT_MANIFEST=$SPLIT_MANIFEST \
            EVALKIT_REV=$EVALKIT_REV \
            EVALKIT_SOURCE_SHA256=$EVALKIT_SOURCE_SHA256 \
            EASYOCR_ROOT=$EASYOCR_ROOT \
            EASYOCR_SOURCE_MODELS=$EASYOCR_SOURCE_MODELS \
            WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED=1 \
            GENERATION_MODE=$mode \
            CPS_NOISE_LEVEL=$level \
            ODE_SOLVER=$solver \
            HEIGHT=512 \
            WIDTH=512 \
            NUM_GPUS=2 \
            CUDA_DEVICES=$devices \
            PREP_WORKERS=4 \
            SCORE_WORKERS=2 \
            SCORE_THREADS_PER_WORKER=8 \
            fish $launcher >$log_path 2>&1 &
        set -a running_pids $last_pid
        set -a running_tasks $wave_tasks[$slot]
        set -a running_logs $log_path
    end

    set -l failed 0
    for slot in (seq (count $running_pids))
        wait $running_pids[$slot]
        set -l rc $status
        set -l task_parts (string split , -- $running_tasks[$slot])
        set -l model_id $task_parts[1]
        set -l sampler_id $task_parts[2]
        if test $rc -ne 0; or not _task_complete $model_id $sampler_id
            set failed 1
            echo "[error] $model_id/$sampler_id failed strict fast audit; log=$running_logs[$slot]" >&2
            tail -n 160 $running_logs[$slot] >&2
        else
            echo "[done]  "(date --iso-8601=seconds)" $model_id/$sampler_id"
        end
    end
    test $failed -eq 0
end

echo "[matrix] models: $model_ids"
echo "[matrix] samplers: $sampler_ids"
echo "[matrix] selected tasks: "(count $tasks)
echo "[matrix] contract: 30 steps, CFG 1.0, seed 0, 512x512x81, 16 FPS, e140 main_v2"
echo "[matrix] output base: $_matrix_output_base"

set -l task_start 1
while test $task_start -le (count $tasks)
    set -l wave
    for offset in (seq 0 (math (count $_matrix_device_pairs) - 1))
        set -l position (math $task_start + $offset)
        if test $position -le (count $tasks)
            set -a wave $tasks[$position]
        end
    end
    echo "[wave] $wave"
    _launch_wave $wave; or _fail "matrix wave failed: $wave"
    set task_start (math $task_start + (count $_matrix_device_pairs))
end

echo "[done] all selected sampler-matrix runs passed the recorded-contract audit"
