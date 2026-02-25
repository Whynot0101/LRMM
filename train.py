#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import io
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from utils.val_tools import * 
import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
import torchvision
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from stable_diffusion_3_reward_pipeline import StableDiffusion3RewardPipeline
from sd3_reward import SD3Transformer2DRewardModel
import copy
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import compute_density_for_timestep_sampling, free_memory
from diffusers.utils.torch_utils import is_compiled_module
if is_wandb_available():
    import wandb

from torchvision.transforms.functional import crop
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
}

        
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        required=False,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=2000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--validation_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", default=False, help="Whether or not to use adafactor (should save mem)"
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--hpdv3_candidate_models",
        nargs="+",
        default=["flux", "kolors", "sd3", "hunyuan", "real_images"],
        help="Candidate models for HPDV3",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )

    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever",
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    # reward settings
    parser.add_argument(
        "--choice_model", type=str, default=None, help="Reward model"
    )
    parser.add_argument(
        "--reward_layer", type=int, default=11, help="Reward layer"
    )
    parser.add_argument(
        "--rm_head_type", type=str, default='linear', help="Reward head type"
    )
    parser.add_argument(
        "--reward_token", type=str, default='average', help="Reward token"
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

            
    return args

def load_text_encoders(args, class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# encode text
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    # print(tokenizer.decode(text_input_ids[0][0]))
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def main():
    
    args = parse_args()
    
    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.validation_steps is not None:
                os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    
    print(args.pretrained_model_name_or_path)
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )
    
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    config = SD3Transformer2DModel.load_config(args.pretrained_model_name_or_path, subfolder="transformer")
    transformer = SD3Transformer2DRewardModel(**config, rm_head_type=args.rm_head_type, reward_token=args.reward_token, reward_layer=args.reward_layer)

    original_model = SD3Transformer2DModel.from_pretrained(
    args.pretrained_model_name_or_path, 
    subfolder="transformer"
    )
    original_state_dict = original_model.state_dict()
    missing_keys, unexpected_keys = transformer.load_state_dict(original_state_dict, strict=False)

    print(f"新增的层（未加载权重）: {missing_keys}")

    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    # gather tokenizers and text encoders
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    # val pipeline 
    val_pipeline = StableDiffusion3RewardPipeline.from_pretrained(args.pretrained_model_name_or_path, 
                    text_encoder=text_encoder_one,
                    text_encoder_2=text_encoder_two,
                    text_encoder_3=text_encoder_three,
                    vae=vae,
                    transformer=transformer, 
                    max_pixels = args.max_pixels,
                    dtype= weight_dtype)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), SD3Transformer2DRewardModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), SD3Transformer2DRewardModel):
                load_model = SD3Transformer2DRewardModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate}

    
    params_to_optimize = [transformer_parameters_with_lr]
    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    
    #DATA
    # transform
    class ImageCropAndResize:
        def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1,
                    max_side=3072): 
            self.height = height
            self.width = width
            self.max_pixels = max_pixels
            self.max_side = max_side  # <-- 修改1：保存 max_side
            self.height_division_factor = height_division_factor
            self.width_division_factor = width_division_factor
            self.final_transform = transforms.Compose(
                [
                    transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        def crop_and_resize(self, image, target_height, target_width):
            width, height = image.size
            scale = max(target_width / width, target_height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height * scale), round(width * scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
            return image

        def get_height_width(self, image):
            if self.height is None or self.width is None:
                width, height = image.size

                if self.max_pixels is not None and width * height > self.max_pixels:
                    scale = (width * height / self.max_pixels) ** 0.5
                    height, width = int(height / scale), int(width / scale)

                if self.max_side is not None:
                    long_side = max(width, height)
                    if long_side > self.max_side:
                        scale = long_side / self.max_side
                        height, width = int(height / scale), int(width / scale)

                height = height // self.height_division_factor * self.height_division_factor
                width = width // self.width_division_factor * self.width_division_factor
            else:
                height, width = self.height, self.width
            return height, width

        def __call__(self, data: Image.Image):
            image = self.crop_and_resize(data, *self.get_height_width(data))
            return self.final_transform(image)

        
    train_transforms = ImageCropAndResize(height=args.resolution, width=args.resolution, max_pixels=args.max_pixels)

    train_dataset = load_dataset(
        "json",
        data_files={
            'train': os.path.join(args.train_data_dir, "train_all.json"),
        },
        cache_dir=args.cache_dir,
    )['train']
    # print(len(train_dataset))
    test_dataset = load_dataset(
        "json",
        data_files={
            'hpdv3_test': os.path.join(args.train_data_dir, "HPDv3_test.json"),
            'hpdv2_test': os.path.join(args.train_data_dir, "HPDv2_test.json"),
            'imagereward_test': os.path.join(args.train_data_dir, "ImageRewardDB_test.json"),
            'pickapic_test': os.path.join(args.train_data_dir, "Pickapic_test.json"),
        },
        cache_dir=args.cache_dir,
    )


    def preprocess_train(examples):

        win_images = [Image.open(os.path.join(args.train_data_dir, path)).convert("RGB")
                        for path in examples['path1']]
        lose_images = [Image.open(os.path.join(args.train_data_dir, path)).convert("RGB")
                        for path in examples['path2']]
        win_pixel_values = [train_transforms(image) for image in win_images]
        lose_pixel_values = [train_transforms(image) for image in lose_images]
    
        examples["win_pixel_values"] = win_pixel_values
        examples["lose_pixel_values"] = lose_pixel_values

        return examples

    def preprocess_test(examples):
        processed_images = []

        for path_list in examples['images']:
            imgs = [
                Image.open(os.path.join(args.train_data_dir, path)).convert("RGB")
                for path in path_list
            ]
            processed_images.append(imgs)

        examples['images'] = processed_images
        return examples

    if args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
    if args.max_val_samples is not None:
        for dataset_name in test_dataset:
            if len(test_dataset[dataset_name]) > args.max_val_samples:
                test_dataset[dataset_name] = test_dataset[dataset_name].shuffle(seed=args.seed).select(range(args.max_val_samples))
    
    train_dataset = train_dataset.with_transform(preprocess_train)
    test_dataset['hpdv3_test'] = test_dataset['hpdv3_test'].with_transform(preprocess_test)
    test_dataset['hpdv2_test'] = test_dataset['hpdv2_test'].with_transform(preprocess_test)
    test_dataset['imagereward_test'] = test_dataset['imagereward_test'].with_transform(preprocess_test)
    test_dataset['pickapic_test'] = test_dataset['pickapic_test'].with_transform(preprocess_test)

    # collate_fn
    def collate_fn(examples):
        win_pixel_values = torch.stack([example["win_pixel_values"] for example in examples])
        lose_pixel_values = torch.stack([example["lose_pixel_values"] for example in examples])
        win_pixel_values = win_pixel_values.to(memory_format=torch.contiguous_format).float()
        lose_pixel_values = lose_pixel_values.to(memory_format=torch.contiguous_format).float()
        return_d =  {"win_pixel_values": win_pixel_values, "lose_pixel_values": lose_pixel_values}
        return_d["caption"] = [example["prompt"] for example in examples]
        
        return return_d

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )

    hpdv3_test_dataloader = torch.utils.data.DataLoader(
        test_dataset['hpdv3_test'],
        shuffle=False,
        collate_fn=lambda x: x,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        drop_last=False
    )
    hpdv2_test_dataloader = torch.utils.data.DataLoader(
        test_dataset['hpdv2_test'],
        shuffle=False,
        collate_fn=lambda x: x,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        drop_last=False
    )
    pickapic_test_dataloader = torch.utils.data.DataLoader(
        test_dataset['pickapic_test'],
        shuffle=False,
        collate_fn=lambda x: x,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        drop_last=False
    )
    imagereward_test_dataloader = torch.utils.data.DataLoader(
        test_dataset['imagereward_test'],
        shuffle=False,
        collate_fn=lambda x: x,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        drop_last=False
    )

    ##### END BIG OLD DATASET BLOCK #####
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    (   hpdv3_test_dataloader, 
        hpdv2_test_dataloader, 
        pickapic_test_dataloader, 
        imagereward_test_dataloader 
    ) = accelerator.prepare(
        hpdv3_test_dataloader, 
        hpdv2_test_dataloader, 
        pickapic_test_dataloader, 
        imagereward_test_dataloader
    )
        
    ### END ACCELERATOR PREP ###
    
    # print(len(train_dataloader))
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config['hpdv3_candidate_models'] = " ".join(args.hpdv3_candidate_models)
        # breakpoint()
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def run_validation(global_step):
        logger.info("***** Running validation *****")
        transformer.eval() 

        total_correct_all = 0
        total_samples_all = 0
        val_results = {}

        with torch.no_grad():
            val_dataloaders = {
                'hpdv3': hpdv3_test_dataloader, 
                'hpdv2': hpdv2_test_dataloader, 
                'pickapic': pickapic_test_dataloader, 
                'imagereward': imagereward_test_dataloader
            }
            
            for name, val_dataloader in val_dataloaders.items():
                local_sum = 0.0
                local_count = 0
            
                pbar = tqdm(val_dataloader, desc=f"Validating {name}", disable=not accelerator.is_local_main_process)
                
                for batch in pbar:
                    for item in batch:
                        prompt = item['prompt']
                        scores = [val_pipeline(prompt, img)[0] for img in item['images']]
                        pred_rank = scores_to_rankvec(scores, higher_is_better=True)
                        local_sum += inversion_score(pred_rank, item['rank'])
                        local_count += 1
                
                gathered_sum = accelerator.gather(torch.tensor(local_sum, device=accelerator.device)).sum().item()
                gathered_count = accelerator.gather(torch.tensor(local_count, device=accelerator.device)).sum().item()
                
                if gathered_count > 0:
                    dataset_acc = gathered_sum / gathered_count
                    val_results[f"val_acc_{name}"] = dataset_acc
                    total_correct_all += gathered_sum
                    total_samples_all += gathered_count
                    logger.info(f"Dataset {name} - Acc: {dataset_acc:.4f}")

            final_acc = total_correct_all / total_samples_all if total_samples_all > 0 else 0
            logger.info(f"Step {global_step}: Overall Val Acc: {final_acc:.4f}")
            log_data = {"val_acc_overall": final_acc}
            log_data.update(val_results) 
            
            accelerator.log(log_data, step=global_step)
        
        transformer.train() 
        return final_acc

    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        score_win_accumulated = 0.0
        score_lose_accumulated = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and (not args.hard_skip_resume):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Dummy processing step {step}, will start training at {resume_step}")
                continue
            # time.sleep(0.1)
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                win_pixel_values = batch["win_pixel_values"]
                lose_pixel_values = batch["lose_pixel_values"]
                prompts = batch["caption"]
                #### Diffusion Stuff ####
                # Convert images to latent space
                with torch.no_grad():
                    # encode prompts
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                                text_encoders, tokenizers, prompts, args.max_sequence_length,
                            )
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                    # encode latents
                    win_pixel_values = win_pixel_values.to(accelerator.device, dtype=weight_dtype)
                    lose_pixel_values = lose_pixel_values.to(accelerator.device, dtype=weight_dtype)
                    win_model_input = vae.encode(win_pixel_values).latent_dist.sample()
                    lose_model_input = vae.encode(lose_pixel_values).latent_dist.sample()
                    win_model_input = (win_model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    lose_model_input = (lose_model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    win_model_input = win_model_input.to(dtype=weight_dtype)
                    lose_model_input = lose_model_input.to(dtype=weight_dtype)
            
                # Sample noise that we'll add to the latents
                win_noise = torch.randn_like(win_model_input)
                lose_noise = torch.randn_like(lose_model_input)
                bsz = win_model_input.shape[0]
                
                # We do not use non-uniform sampling

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                # u = compute_density_for_timestep_sampling(
                #     weighting_scheme=args.weighting_scheme,
                #     batch_size=bsz,
                #     logit_mean=args.logit_mean,
                #     logit_std=args.logit_std,
                #     mode_scale=args.mode_scale,
                # )
                # indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                indices = torch.randint(0, noise_scheduler_copy.config.num_train_timesteps, (bsz,)).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=win_model_input.device)
                
                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                
                sigmas = get_sigmas(timesteps, n_dim=win_model_input.ndim, dtype=win_model_input.dtype)
                win_noisy_model_input = (1.0 - sigmas) * win_model_input + sigmas * win_noise
                lose_noisy_model_input = (1.0 - sigmas) * lose_model_input + sigmas * lose_noise
                # Predict the noise residual

                score_win = transformer(
                    hidden_states=win_noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False
                )[0]
                score_lose = transformer(
                    hidden_states=lose_noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False
                )[0]

                # BT loss
                inside_term = (score_win - score_lose)
                loss = -1 * torch.nn.functional.logsigmoid(inside_term).mean()

                # Gather for logging
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                # Gather the losses across all processes for logging 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - Implicit accuracy
                avg_acc = accelerator.gather(implicit_acc).mean().item()
                # - score_win
                avg_score_win = accelerator.gather(score_win.mean().repeat(args.train_batch_size)).mean()
                avg_score_lose = accelerator.gather(score_lose.mean().repeat(args.train_batch_size)).mean()
                implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps
                score_win_accumulated += avg_score_win / args.gradient_accumulation_steps
                score_lose_accumulated += avg_score_lose / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                accelerator.log({"score_win_accumulated": score_win_accumulated}, step=global_step)
                accelerator.log({"score_lose_accumulated": score_lose_accumulated}, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0
                score_win_accumulated = 0.0
                score_lose_accumulated = 0.0
                
                if global_step % args.validation_steps == 0:
                    run_validation(global_step)
                    
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info("Pretty sure saving/loading is fixed but proceed cautiously")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        pipeline = StableDiffusion3RewardPipeline.from_pretrained(
            args.pretrained_model_name_or_path, transformer=transformer
        )
        # save the pipeline
        pipeline.save_pretrained(args.output_dir)
    


    accelerator.end_training()


if __name__ == "__main__":
    main()