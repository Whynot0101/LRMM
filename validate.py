import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from datasets import Dataset, load_dataset
from sd3_reward import SD3Transformer2DRewardModel
from stable_diffusion_3_reward_pipeline import StableDiffusion3RewardPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Validate LRMM-SD3 reward model.")
    parser.add_argument("--transformer_path", type=str, default="whynot0128/LRMM-SD3", help="Path to the transformer model.")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers", help="Path to the base SD3 diffusers model.")
    parser.add_argument("--train_data_dir", type=str, default="data/HPDv3", help="Directory containing the validation dataset.")
    parser.add_argument("--cache_dir", type=str, default="./huggingface_cache/datasets", help="Huggingface dataset cache directory.")
    parser.add_argument("--timestep", type=int, default=1, help="Timestep to use for pipeline.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # pipeline
    transformer = SD3Transformer2DRewardModel.from_pretrained(args.transformer_path, torch_dtype=torch.bfloat16)
    pipeline = StableDiffusion3RewardPipeline.from_pretrained(args.model_path, transformer=transformer, torch_dtype=torch.bfloat16)
    pipeline.to("cuda")

    # dataset
    data_files = {}
    data_files['test'] = os.path.join(args.train_data_dir, "HPDv3_test.json")
    dataset = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=args.cache_dir,
    )['test']

    acc = 0
    acc_tie = 0
    num_rows = len(dataset)

    for data in tqdm(dataset):
        # Use the unified "images" list generated in HPDv3_test.json
        # The first image is expected to be the winner (index 0) and the second the loser (index 1) based on HPDv3 structure
        image_w = Image.open(os.path.join("data", data['images'][0])).convert("RGB")
        image_l = Image.open(os.path.join("data", data['images'][1])).convert("RGB")
        images = [image_w, image_l]
        prompt = [data['prompt']] 
        score_w = pipeline(prompt, images=image_w, timestep=args.timestep)[0]
        score_l = pipeline(prompt, images=image_l, timestep=args.timestep)[0]
        acc += 1 if score_w>score_l else 0
        acc_tie += 1 if score_w>=score_l else 0

    acc_rate = acc / num_rows
    tie_rate = acc_tie / num_rows

    print("\n" + "="*50)
    print(f"| {'Metric':<20} | {'Value':<23} |")
    print("-" * 50)
    print(f"| {'Accuracy':<20} | {acc_rate:<23.4f} |")
    print(f"| {'Tie Rate':<20} | {tie_rate:<23.4f} |")
    print("="*50 + "\n")

    results = {
        "dataset": args.train_data_dir,
        "accuracy": acc_rate,
        "tie_rate": tie_rate,
        "num_samples": num_rows
    }
    
    import json
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to validation_results.json")

if __name__ == "__main__":
    main()
