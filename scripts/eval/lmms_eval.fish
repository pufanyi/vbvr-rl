#!/usr/bin/env fish
# Full VBVR-Bench eval for Wan2.2-I2V-A14B.
# Defaults target 8x H100, 50 steps x 81 frames; override DATA_PARALLEL,
# NUM_GPUS, resolution, frames, or steps from the environment as needed.
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

set -q DATA_PARALLEL[1]; or set DATA_PARALLEL 8
set -q NUM_GPUS[1]; or set NUM_GPUS 1
set -q SP_SIZE[1]; or set SP_SIZE 1
set -q TP_SIZE[1]; or set TP_SIZE 1
set -q NUM_INFERENCE_STEPS[1]; or set NUM_INFERENCE_STEPS 50
set -q NUM_FRAMES[1]; or set NUM_FRAMES 81
set -q HEIGHT[1]; or set HEIGHT 384
set -q WIDTH[1]; or set WIDTH 384
set -q FPS[1]; or set FPS 16
set -q ENABLE_TORCH_COMPILE[1]; or set ENABLE_TORCH_COMPILE True

set -l MODEL_NAME (basename $MODEL_DIR)
set -l GENERATED_VIDEO_DIR $OUTPUT_DIR/generated_videos/$MODEL_NAME
mkdir -p $GENERATED_VIDEO_DIR

set MODEL_ARGS "model=$MODEL_DIR"
set MODEL_ARGS "$MODEL_ARGS,output_dir=$GENERATED_VIDEO_DIR"
set MODEL_ARGS "$MODEL_ARGS,data_parallel=$DATA_PARALLEL,num_gpus=$NUM_GPUS,sp_size=$SP_SIZE,tp_size=$TP_SIZE"
set MODEL_ARGS "$MODEL_ARGS,num_inference_steps=$NUM_INFERENCE_STEPS,num_frames=$NUM_FRAMES"
set MODEL_ARGS "$MODEL_ARGS,height=$HEIGHT,width=$WIDTH,fps=$FPS"
set MODEL_ARGS "$MODEL_ARGS,dit_cpu_offload=False,text_encoder_cpu_offload=True"
set MODEL_ARGS "$MODEL_ARGS,image_encoder_cpu_offload=False,vae_cpu_offload=False"
set MODEL_ARGS "$MODEL_ARGS,enable_torch_compile=$ENABLE_TORCH_COMPILE"

set -l MODEL_DIR_LOWER (string lower -- $MODEL_DIR)
if string match -q '*5b*' -- $MODEL_DIR_LOWER
    set MODEL_ARGS "$MODEL_ARGS,override_pipeline_cls_name=WanPipeline,ti2v_task=True,flow_shift=5"
end

exec stdbuf -oL -eL .venv/bin/python -m lmms_eval eval \
    --model fastvideo \
    --model_args $MODEL_ARGS \
    --tasks vbvr \
    --batch_size 1 \
    --log_samples \
    --output_path=$OUTPUT_DIR
