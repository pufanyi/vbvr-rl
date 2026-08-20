#!/usr/bin/env fish
# Install the isolated Pixi vLLM environment used by the Qwen3.6 judge.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

set -l runtime_root (set -q WAN_TRAINER_VLLM_PROVENANCE_DIR; and echo $WAN_TRAINER_VLLM_PROVENANCE_DIR; or echo storage/host_vllm)

if not type -q pixi
    echo "ERROR: Pixi is required to create the isolated vLLM environment." >&2
    exit 1
end

mkdir -p $runtime_root
echo "Installing the locked, isolated Pixi 'vllm' environment..."
pixi install --environment vllm --locked; or exit 1

pixi run --environment vllm --locked python -c \
    'import ray, torch, transformers, vllm; print(f"vllm={vllm.__version__} ray={ray.__version__} torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'
or exit 1
pixi run --environment vllm --locked ninja --version >/dev/null
or begin
    echo "ERROR: vLLM JIT compiler helper 'ninja' is missing from the Pixi vllm environment." >&2
    exit 1
end

pixi list --environment vllm --locked --json >"$runtime_root/pixi-packages.json"; or exit 1

echo "Standalone Pixi 'vllm' environment is ready."
echo "The exact resolution is recorded in pixi.lock."
