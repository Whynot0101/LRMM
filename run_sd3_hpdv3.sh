#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 固定参数
MODEL_NAME="../stable-diffusion-3"
DATASET_NAME="../" # make sure "pickapic" in dataset_name
WORLD_SIZE=8
ACCUMULATION_STEPS=16
BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
CACHE_DIR="../huggingface_cache/datasets"
MAX_TRAIN_STEPS=60000
MIXED_PRECISION="bf16"
DATALOADER_WORKERS=8
LR=5e-6


RUN_NAME="All_lr${LR}_bs${BATCH_SIZE}_special_mlp_layer11_all_size512_rewardzero"
OUTPUT_DIR="../output/SD3_Reward/${RUN_NAME}"

accelerate launch train.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_batch_size=1 \
  --split="train" \
  --mixed_precision="$MIXED_PRECISION" \
  --dataloader_num_workers="$DATALOADER_WORKERS" \
  --gradient_accumulation_steps="$ACCUMULATION_STEPS" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --max_pixels=262144 \
  --learning_rate="$LR" \
  --cache_dir="$CACHE_DIR" \
  --checkpointing_steps 2000 \
  --validation_steps 200 \
  --reward_layer 11 \
  --output_dir="$OUTPUT_DIR" \
  --tracker_project_name="$RUN_NAME" \
  --reward_token special \
  --rm_head_type mlp \

