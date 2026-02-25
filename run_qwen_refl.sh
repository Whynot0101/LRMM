#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 固定参数
MODEL_NAME="/mnt/ramdisk/qwen"
REWARD_NAME='/mnt/ramdisk/qwen_reward/'
DATASET_NAME="/zjk_nas/zhiyi/data/HPDv3" # make sure "pickapic" in dataset_name
WORLD_SIZE=8
ACCUMULATION_STEPS=2
BATCH_SIZE=$((WORLD_SIZE * ACCUMULATION_STEPS))
CACHE_DIR="/zjk_nas/zhiyi/huggingface_cache/datasets"
MAX_TRAIN_STEPS=15000
MIXED_PRECISION="bf16"
DATALOADER_WORKERS=8
LR=1e-5
RANK=32
ALPHA=32

RUN_NAME="lr${LR}_bs${BATCH_SIZE}_rank${RANK}_alpha${ALPHA}"
OUTPUT_DIR="/zjk_nas/zhiyi/output/QwenImage_ReFL/${RUN_NAME}"

accelerate launch train_qwenimage_ReFL.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --reward_model_name_or_path="$REWARD_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_batch_size=1 \
  --split="train_filtered" \
  --mixed_precision="$MIXED_PRECISION" \
  --dataloader_num_workers="$DATALOADER_WORKERS" \
  --gradient_accumulation_steps="$ACCUMULATION_STEPS" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=0 \
  --resolution=1024 \
  --max_train_samples 24\
  --learning_rate="$LR" \
  --cache_dir="$CACHE_DIR" \
  --checkpointing_steps 1000 \
  --validation_steps 10 \
  --output_dir="$OUTPUT_DIR" \
  --tracker_project_name="$RUN_NAME" \
