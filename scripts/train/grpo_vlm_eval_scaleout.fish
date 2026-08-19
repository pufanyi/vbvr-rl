#!/usr/bin/env fish

source (dirname (status filename))/../lib/env.fish

# Require the experiment YAML explicitly so the submitted command records the
# exact training contract. Other arguments are forwarded as CLI overrides.
set -l config ""
set -l train_args
set -l has_output_dir false
set -l has_wandb_run_name false
for arg in $argv
    if string match -q -- '--yaml=*' "$arg"
        if test -n "$config"
            echo "ERROR: --yaml may be specified only once" >&2
            exit 1
        end
        set config (string replace -- '--yaml=' '' "$arg")
        if test -z "$config"
            echo "ERROR: --yaml requires a nonempty path" >&2
            exit 1
        end
    else if test "$arg" = "--yaml"
        echo "ERROR: use --yaml=<path> (with '=')" >&2
        exit 1
    else if test "$arg" = "--config"; or string match -q -- '--config=*' "$arg"
        echo "ERROR: this wrapper accepts --yaml=<path>; do not also pass --config" >&2
        exit 1
    else
        set -a train_args "$arg"
        if test "$arg" = "--output_dir"; or string match -q -- '--output_dir=*' "$arg"
            set has_output_dir true
        end
        if test "$arg" = "--wandb_run_name"; or string match -q -- '--wandb_run_name=*' "$arg"
            set has_wandb_run_name true
        end
    end
end

if test -z "$config"
    echo "ERROR: missing required --yaml=<path>" >&2
    echo "Usage: fish scripts/train/grpo_vlm_eval_scaleout.fish --yaml=configs/experiment.yaml [training overrides...]" >&2
    exit 1
end
if not test -f "$config"
    echo "ERROR: YAML config does not exist: $config" >&2
    exit 1
end
set config (realpath "$config")

# WORLD_SIZE is the scheduler machine count. The bs32/G32 contract requires
# wave16 and a data-parallel world that satisfies the prompt-group divisibility rules.
if not set -q MASTER_ADDR; or test -z "$MASTER_ADDR"
    echo "ERROR: MASTER_ADDR is not set" >&2
    exit 1
end
if not set -q WORLD_SIZE; or not contains -- "$WORLD_SIZE" 4 8 16
    echo "ERROR: WORLD_SIZE must be the node count 4, 8, or 16; got '$WORLD_SIZE'" >&2
    exit 1
end
if not set -q RANK; or not string match -qr '^[0-9]+$' -- "$RANK"; or test "$RANK" -ge "$WORLD_SIZE"
    echo "ERROR: RANK must be an integer in [0, WORLD_SIZE), got '$RANK' for WORLD_SIZE=$WORLD_SIZE" >&2
    exit 1
end
set -q MASTER_PORT; or set -gx MASTER_PORT 29500

set -l nproc 8
set -l global_world_size (math "$WORLD_SIZE * $nproc")
set -l topology_suffix (string join '' _nodes $WORLD_SIZE _world $global_world_size)

# Four TP2 Qwen replicas are co-hosted on each node. Internal vLLM DP dispatch
# balances each node's requests across the four engines. These defaults can be
# overridden before invoking this wrapper.
set -q WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE; or set -gx WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE 2
set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE 4
set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL 4
set -q WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND mp
set -q WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND; or set -gx WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND mp
set -q WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION; or set -gx WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION 0.50
set -q WAN_TRAINER_VLM_LOG_DIR; or set -gx WAN_TRAINER_VLM_LOG_DIR "/tmp/wan-trainer-vllm-world$global_world_size"
set -q WAN_TRAINER_KERNELS_CACHE; or set -gx WAN_TRAINER_KERNELS_CACHE ~/.cache/wan-trainer/kernels

if not test -d "$WAN_TRAINER_KERNELS_CACHE"
    echo "ERROR: pinned FA3 cache is missing: $WAN_TRAINER_KERNELS_CACHE" >&2
    echo "Prefetch it before submitting the multi-node job." >&2
    exit 1
end

# Keep checkpoint and W&B namespaces isolated when the same YAML is tried at
# 4, 8, and 16 nodes. Explicit CLI overrides still win.
set -l topology_args
if test "$has_output_dir" = false
    set -l config_output_dir (.venv/bin/python -c \
        'import sys, yaml; print((yaml.safe_load(open(sys.argv[1])) or {}).get("output_dir", ""))' \
        "$config")
    or exit 1
    if test -n "$config_output_dir"
        set -a topology_args --output_dir "$config_output_dir$topology_suffix"
    end
end
if test "$has_wandb_run_name" = false
    set -l config_wandb_name (.venv/bin/python -c \
        'import sys, yaml; print((yaml.safe_load(open(sys.argv[1])) or {}).get("wandb_run_name", "") or "")' \
        "$config")
    or exit 1
    if test -n "$config_wandb_name"
        set -a topology_args --wandb_run_name "$config_wandb_name$topology_suffix"
    end
end

echo "Launching VLM training: node=$RANK/$WORLD_SIZE global_world=$global_world_size master=$MASTER_ADDR:$MASTER_PORT"
echo "Config: $config"
echo "Qwen topology per node: DP$WAN_TRAINER_VLM_DATA_PARALLEL_SIZE x TP$WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE; memory budget: $WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION; FA3 cache: $WAN_TRAINER_KERNELS_CACHE"

# Extra CLI arguments are appended last and therefore override both YAML and
# topology-derived output names.
if set -q WAN_TRAINER_VLM_LAUNCH_DRY_RUN; and test "$WAN_TRAINER_VLM_LAUNCH_DRY_RUN" = 1
    echo "[dry-run] topology overrides: "(string join ' ' -- (string escape -- $topology_args))
    echo "[dry-run] user overrides: "(string join ' ' -- (string escape -- $train_args))
    exit 0
end

exec fish scripts/train/grpo_vlm_eval_multinode.fish \
    --nproc $nproc \
    --config $config \
    $topology_args \
    $train_args
