import os, io
import argparse
from PIL import Image
from tqdm import tqdm
from stable_diffusion_3_reward_pipeline import StableDiffusion3RewardPipeline
from sd3_reward import SD3Transformer2DRewardModel
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset

# -------------------------
# metrics helpers
# -------------------------
def scores_to_rankvec(scores, higher_is_better=True):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=higher_is_better)
    rank = [0] * len(scores)
    for r, idx in enumerate(order):
        rank[idx] = r
    return rank

def inversion_score(p1, p2):
    assert len(p1) == len(p2), f"{len(p1)}, {len(p2)}"
    n = len(p1)
    if n < 2:
        return 1.0
    cnt = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (p1[i] >= p1[j] and p2[i] < p2[j]) or (p1[i] < p1[j] and p2[i] > p2[j]):
                cnt += 1
    return 1 - cnt / (n * (n - 1) / 2)

# -------------------------
# collate (因为有 PIL.Image，默认 collate 会失败)
# -------------------------
def collate_fn(batch):
    images = [x["images"] for x in batch]
    prompts = [x["prompt"] for x in batch]
    ranks = [x["rank"] for x in batch]
    return {"images": images, "prompts": prompts, "ranks": ranks}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reward models on datasets.")
    parser.add_argument("--dataset_name", nargs="+", default=["HPDv2", "HPDv3", "Pick", "ImageRewardDB"], 
                        help="List of dataset names (e.g. HPDv2 HPDv3 Pick ImageRewardDB)")
    parser.add_argument("--scorers", nargs="+", default=["LRMM-SD3", "pickscore", "hpsv2", "aesthetic", "clip", "imagereward", "hpsv3"], 
                        help="List of scorers to evaluate (e.g. LRMM-SD3 pickscore hpsv2 aesthetic clip imagereward hpsv3)")
    parser.add_argument("--transformer_path", type=str, default="whynot0128/LRMM-SD3")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--cache_dir", type=str, default="./huggingface_cache/datasets")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # -------------------------
    # init scorers
    # -------------------------
    scorers = {}
    
    if "LRMM-SD3" in args.scorers:
        transformer = SD3Transformer2DRewardModel.from_pretrained(args.transformer_path, torch_dtype=torch.bfloat16)
        scorers["LRMM-SD3"] = StableDiffusion3RewardPipeline.from_pretrained(args.model_path, transformer=transformer, torch_dtype=torch.bfloat16).to(device)

    scorer_mapping = {
        "pickscore": "PickScoreScorer",
        "hpsv2": "HPSv2",
        "aesthetic": "AestheticScorer",
        "clip": "ClipScorer",
        "imagereward": "ImageRewardScorer",
        "HPS": "HPS",
        "MPS": "MPS",
    }

    if any(s in args.scorers for s in scorer_mapping) or "hpsv3" in args.scorers:
        import rewards as reward_models

    for s_name in args.scorers:
        if s_name in scorer_mapping:
            scorer_cls = getattr(reward_models, scorer_mapping[s_name])
            scorers[s_name] = scorer_cls(device)
        elif s_name == "hpsv3":
            scorers["hpsv3"] = reward_models.HPSv3Scorer(device, api_url="http://127.0.0.1:8000/infer_score")

    for dataset_name in args.dataset_name:
        if accelerator.is_main_process:
            print(f"\n========== Evaluating on {dataset_name} ==========")

        # -------------------------
        # load dataset
        # -------------------------
        ds_map = {
            "hpdv2": ("data/HPDv2", "HPDv2_test.json"),
            "hpdv3": ("data/HPDv3", "HPDv3_test.json"),
            "pick": ("data/Pickapic", "Pickapic_test.json"),
            "imagerewarddb": ("data/ImageRewardDB", "ImageRewardDB_test.json")
        }

        matched_key = next((k for k in ds_map if k in dataset_name.lower()), None)
        if not matched_key:
            raise ValueError(f"Unknown dataset type: {dataset_name}")

        train_data_dir, json_file = ds_map[matched_key]
        data_files = {"test": os.path.join(train_data_dir, json_file)}
        dataset = load_dataset("json", data_files=data_files, cache_dir=args.cache_dir)["test"]

        def preprocess(examples):
            images_batch = [
                [Image.open(os.path.join("data", p)).convert("RGB") for p in paths]
                for paths in examples["images"]
            ]
            return {"images": images_batch, "prompt": examples["prompt"], "rank": examples["rank"]}

        dataset = dataset.with_transform(preprocess)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        dataloader = accelerator.prepare(dataloader)

        # -------------------------
        # eval
        # -------------------------
        local_sum = {name: 0.0 for name in scorers}
        local_cnt = 0

        iterator = dataloader
        if accelerator.is_main_process:
            iterator = tqdm(dataloader, desc=f"eval {dataset_name}")

        for batch in iterator:
            for images, prompt, gt_rank in zip(batch["images"], batch["prompts"], batch["ranks"]):
                for name, scorer in scorers.items():
                    scores = [scorer(prompt, img)[0] if isinstance(scorer(prompt, img), list) else scorer(prompt, img) for img in images]
                    pred_rank = scores_to_rankvec(scores, higher_is_better=True)
                    local_sum[name] += inversion_score(pred_rank, gt_rank)
                local_cnt += 1

        # -------------------------
        # all-reduce 汇总
        # -------------------------
        names = list(scorers.keys())
        sum_tensor = torch.tensor([local_sum[n] for n in names], device=device, dtype=torch.float64)
        cnt_tensor = torch.tensor([local_cnt], device=device, dtype=torch.float64)

        sum_tensor = accelerator.reduce(sum_tensor, reduction="sum")
        cnt_tensor = accelerator.reduce(cnt_tensor, reduction="sum")

        if accelerator.is_main_process:
            acc = {n: (sum_tensor[i].item() / cnt_tensor.item()) for i, n in enumerate(names)}
            
            print("\n" + "="*50)
            print(f"| Dataset: {dataset_name:<36} |")
            print("-" * 50)
            print(f"| {'Model':<20} | {'Accuracy':<23} |")
            print("-" * 50)
            for n, a in acc.items():
                print(f"| {n:<20} | {a:<23.4f} |")
            print("="*50 + "\n")
            
            # Save results to JSON
            results_file = "benchmark_results.json"
            if os.path.exists(results_file):
                import json
                with open(results_file, "r") as f:
                    all_results = json.load(f)
            else:
                all_results = {}
            
            all_results[dataset_name] = acc
            with open(results_file, "w") as f:
                import json
                json.dump(all_results, f, indent=4)
            print(f"Results appended to {results_file}")

if __name__ == "__main__":
    main()
