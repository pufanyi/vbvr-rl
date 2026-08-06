#!/usr/bin/env fish
# Build the isolated vLLM runtime used by the standalone Qwen3.6 judge.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

set -l runtime_root (set -q WAN_TRAINER_HOST_VLLM_ROOT; and echo $WAN_TRAINER_HOST_VLLM_ROOT; or echo storage/host_vllm)
set -l venv "$runtime_root/.venv"
set -l uv_cache (set -q WAN_TRAINER_HOST_VLLM_UV_CACHE; and echo $WAN_TRAINER_HOST_VLLM_UV_CACHE; or echo /tmp/wan-trainer-host-vllm-uv-cache)
set -l vllm_version (set -q WAN_TRAINER_VLLM_VERSION; and echo $WAN_TRAINER_VLLM_VERSION; or echo 0.26.0)
set -l package_index (set -q WAN_TRAINER_HOST_VLLM_INDEX; and echo $WAN_TRAINER_HOST_VLLM_INDEX; or echo https://pypi.tuna.tsinghua.edu.cn/simple)

if not type -q uv
    echo "ERROR: uv is required to create the isolated vLLM environment." >&2
    exit 1
end

mkdir -p $runtime_root $uv_cache
if not test -x "$venv/bin/python"
    uv venv --python 3.12 $venv
    or exit 1
end

echo "Installing vLLM $vllm_version into $venv (index=$package_index, cache=$uv_cache, proxy disabled)..."
env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    UV_CACHE_DIR=$uv_cache \
    UV_HTTP_TIMEOUT=600 \
    UV_NO_PROGRESS=1 \
    uv pip install \
    --no-config \
    --python "$venv/bin/python" \
    --default-index $package_index \
    --link-mode copy \
    --torch-backend=auto \
    "vllm==$vllm_version" \
    'numpy==2.3.5' \
    'ray[cgraph]'
or exit 1

"$venv/bin/python" -c \
    'import ray, torch, transformers, vllm; print(f"vllm={vllm.__version__} ray={ray.__version__} torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'
or exit 1
test -x "$venv/bin/ninja"
or begin
    echo "ERROR: vLLM JIT compiler helper is missing: $venv/bin/ninja" >&2
    exit 1
end

env UV_CACHE_DIR=$uv_cache uv pip freeze --no-config --python "$venv/bin/python" >"$runtime_root/requirements.freeze.txt"
or exit 1

echo "Standalone vLLM environment is ready: $venv"
