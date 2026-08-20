#!/usr/bin/env fish
# Download and checksum-verify the pinned Qwen3.6-27B judge snapshot.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

set -l repo Qwen/Qwen3.6-27B
set -l revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
set -l output (set -q WAN_TRAINER_VLM_MODEL_PATH; and echo $WAN_TRAINER_VLM_MODEL_PATH; or echo storage/models/Qwen3.6-27B)
set -l workers (set -q WAN_TRAINER_HF_DOWNLOAD_WORKERS; and echo $WAN_TRAINER_HF_DOWNLOAD_WORKERS; or echo 16)

if not command -q pixi
    echo "ERROR: Pixi is required to run the isolated Hugging Face CLI." >&2
    exit 1
end
pixi install --environment vllm --locked; or exit 1

mkdir -p (dirname $output)
echo "Downloading $repo@$revision to $output through hf-mirror.com with $workers workers..."
env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    pixi run --environment vllm --locked hf download $repo \
    --revision $revision \
    --local-dir $output \
    --max-workers $workers
or exit 1

echo "Verifying every remote file and LFS checksum..."
env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    pixi run --environment vllm --locked hf cache verify $repo \
    --revision $revision \
    --local-dir $output \
    --fail-on-missing-files \
    --fail-on-extra-files
