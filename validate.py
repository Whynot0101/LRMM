from sd3_reward import SD3Transformer2DRewardModel
from datasets import Dataset, load_dataset
from stable_diffusion_3_reward_pipeline import StableDiffusion3RewardPipeline
import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

# args
transformer_path='/zjk_nas/zhiyi/output/SD3_Reward/HPDv3_lr5e-6_bs128_special_mlp_layer11_all_size1024/checkpoint-9500/transformer'
model_path='/zjk_nas/zhiyi/output/SD3_Reward/HPDv3_lr5e-6_bs128_special_mlp_layer11_all_size1024'
train_data_dir='/zjk_nas/zhiyi/data/HPDv3'
cache_dir='/zjk_nas/zhiyi/huggingface_cache/datasets'
timestep = 1
# pipeline
transformer = SD3Transformer2DRewardModel.from_pretrained(transformer_path, torch_dtype=torch.bfloat16)
pipeline = StableDiffusion3RewardPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.bfloat16)
pipeline.to("cuda")

# dataset
data_files = {}
data_files['test'] = os.path.join(train_data_dir, "test.json")
dataset = load_dataset(
    "json",
    data_files=data_files,
    cache_dir=cache_dir,
)['test']

acc = 0
acc_tie = 0
num_rows = len(dataset)

for data in tqdm(dataset):
    image_w = Image.open(os.path.join(train_data_dir, data['path1'])).convert("RGB")
    image_l = Image.open(os.path.join(train_data_dir, data['path2'])).convert("RGB")
    images = [image_w, image_l]
    prompt = [data['prompt']] 
    score_w = pipeline(prompt, images=image_w, timestep=timestep)[0]
    score_l = pipeline(prompt, images=image_l, timestep=timestep)[0]
    acc += 1 if score_w>score_l else 0
    acc_tie += 1 if score_w>=score_l else 0

print('acc:', acc/num_rows, 'acc_tie:', acc_tie/num_rows)
