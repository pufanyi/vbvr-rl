#!/usr/bin/env fish
# Full VBVR-Bench eval for Wan2.2-I2V-A14B (8x H100, 50 steps x 81 frames).
# Run from the lmms-eval repo root.

cd /mnt/umm/users/pufanyi/workspace/lmms-eval; or exit 1

# Rule-based VBVR scorers read the GT mp4s/pngs from this root.
set -gx VBVR_GT_PATH /mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/datasets/VBVR-Bench

# set MODEL_DIR   /mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/models/Wan2.2-I2V-A14B-Diffusers
set -q MODEL_DIR[1]; or begin
    echo "[error] MODEL_DIR is required"
    exit 1
end
set -q OUTPUT_DIR[1]; or set -gx OUTPUT_DIR /mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/lmms_eval
set -gx LMMS_EVAL_DATASETS_CACHE /tmp/lmms_eval_hf_datasets_(whoami)
mkdir -p $LMMS_EVAL_DATASETS_CACHE

set -l MODEL_NAME (basename $MODEL_DIR)
set -l GENERATED_VIDEO_DIR $OUTPUT_DIR/generated_videos/$MODEL_NAME
mkdir -p $GENERATED_VIDEO_DIR

set MODEL_ARGS "model=$MODEL_DIR"
set MODEL_ARGS "$MODEL_ARGS,output_dir=$GENERATED_VIDEO_DIR"
set MODEL_ARGS "$MODEL_ARGS,data_parallel=8,num_gpus=1,sp_size=1,tp_size=1"
set MODEL_ARGS "$MODEL_ARGS,num_inference_steps=50,num_frames=81"
set MODEL_ARGS "$MODEL_ARGS,height=384,width=384,fps=16"
set MODEL_ARGS "$MODEL_ARGS,dit_cpu_offload=False,text_encoder_cpu_offload=True"
set MODEL_ARGS "$MODEL_ARGS,image_encoder_cpu_offload=False,vae_cpu_offload=False"
set MODEL_ARGS "$MODEL_ARGS,enable_torch_compile=True"

exec stdbuf -oL -eL .venv/bin/python -m lmms_eval eval \
    --model fastvideo \
    --model_args $MODEL_ARGS \
    --tasks vbvr \
    --batch_size 1 \
    --log_samples \
    --output_path=$OUTPUT_DIR
