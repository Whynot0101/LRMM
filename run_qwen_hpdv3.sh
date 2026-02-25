#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 固定参数
MODEL_NAME="/mnt/ramdisk/Qwen-Image"
DATASET_NAME="../" # make sure "pickapic" in dataset_name
WORLD_SIZE=8
ACCUMULATION_STEPS=8
BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
CACHE_DIR="../huggingface_cache/datasets"
MAX_TRAIN_STEPS=40000
MIXED_PRECISION="bf16"
DATALOADER_WORKERS=8
LR=1e-4


RUN_NAME="HPDv3_lr${LR}_bs${BATCH_SIZE}_special_mlp_layer29_all_layernorm_fullft_1024_inner"
OUTPUT_DIR="../output/QwenImage_Reward/${RUN_NAME}"

accelerate launch train_qwenimage.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_batch_size=1 \
  --split="train" \
  --mixed_precision="$MIXED_PRECISION" \
  --dataloader_num_workers="$DATALOADER_WORKERS" \
  --gradient_accumulation_steps="$ACCUMULATION_STEPS" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=500 \
  --max_pixels=262144 \
  --learning_rate="$LR" \
  --cache_dir="$CACHE_DIR" \
  --checkpointing_steps 1000 \
  --validation_steps 100 \
  --reward_layer 29 \
  --output_dir="$OUTPUT_DIR" \
  --tracker_project_name="$RUN_NAME" \
  --reward_token special \
  --rm_head_type mlp \
  --gradient_checkpointing \

