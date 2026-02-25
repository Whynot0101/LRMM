import os, io
from PIL import Image
from tqdm import tqdm
from stable_diffusion_3_reward_pipeline import StableDiffusion3RewardPipeline
from sd3_reward import SD3Transformer2DRewardModel
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
# from rewards import PickScoreScorer, HPSv2, AestheticScorer, ClipScorer, ImageRewardScorer, HPSv3Scorer, HPS
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
            # if p1[i] == p1[j]:
            #     cnt += 1  # 跳过 pred tie
            if (p1[i] >= p1[j] and p2[i] < p2[j]) or (p1[i] < p1[j] and p2[i] > p2[j]):
                cnt += 1
    return 1 - cnt / (n * (n - 1) / 2)


# -------------------------
# collate (因为有 PIL.Image，默认 collate 会失败)
# -------------------------
def collate_fn(batch):
    # batch: list[dict(images=list[PIL], prompt=str, rank=list[int])]
    images = [x["images"] for x in batch]   # list[list[PIL]]
    prompts = [x["prompt"] for x in batch]  # list[str]
    ranks = [x["rank"] for x in batch]      # list[list[int]]
    return {"images": images, "prompts": prompts, "ranks": ranks}


def main():
    accelerator = Accelerator()
    device = accelerator.device

    # -------------------------
    # scorers (每个进程各自一份)
    # -------------------------
    transformer_path='../transformer'
    model_path='/mnt/fuse/.cache/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671'
    transformer = SD3Transformer2DRewardModel.from_pretrained(transformer_path, torch_dtype=torch.bfloat16)
    pipeline = StableDiffusion3RewardPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
    
    scorers = {
    # "pickscore": PickScoreScorer('cuda'),
    # "hpsv2": HPSv2('cuda'),
    # "aesthetic": AestheticScorer('cuda'),
    # "clip": ClipScorer('cuda'),
    # "imagereward": ImageRewardScorer('cuda'),
    # "hpsv3": HPSv3Scorer('cuda'),
    "LRMM-SD3": pipeline,
    # "HPS": HPS('cuda'),
    # "MPS": MPS('cuda'),
}

    # -------------------------
    # dataset load
    # -------------------------
    # dataset_name = "ImageRewardDB"
    dataset_name = "Pick"
    # dataset_name = "HPDv2"
    # dataset_name = "HPDv3"
    cache_dir = "/zjk_nas/zhiyi/huggingface_cache/datasets"

    # -------------------------
    # preprocess (with_transform: 输入单样本 dict，输出单样本 dict)
    # -------------------------
    if "hpdv2" in dataset_name.lower():
        train_data_dir = '/zjk_nas/zhiyi/data/HPDv2'
        data_files = {"test": os.path.join(train_data_dir, "test.json")}
        dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)["test"]
        def preprocess(examples):
            # examples["image_path"]: list[list[str]]
            images_batch = [
                [Image.open(os.path.join(train_data_dir,"test", p)).convert("RGB") for p in paths]
                for paths in examples["image_path"]
            ]
            return {
                "images": images_batch,
                "prompt": examples["prompt"],
                "rank": examples["rank"],
            }

    elif "hpdv3" in dataset_name.lower():
        train_data_dir = '/zjk_nas/zhiyi/data/HPDv3'
        data_files = {"test": os.path.join(train_data_dir, "test.json")}
        dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)["test"]
        def preprocess(examples):
            # examples["path1"]/["path2"]: list[str]
            images_batch = []
            for p1, p2 in zip(examples["path1"], examples["path2"]):
                images_batch.append([
                    Image.open(os.path.join(train_data_dir, p1)).convert("RGB"),
                    Image.open(os.path.join(train_data_dir, p2)).convert("RGB"),
                ])
            return {
                "images": images_batch,
                "prompt": examples["prompt"],
                "rank": [[0, 1]] * len(images_batch),
            }

    elif "pick" in dataset_name.lower():
        dataset = load_dataset(
            "parquet",
            data_files={
                "test": "/zjk_nas/xinxuan/model/datasets--liuhuohuo2--pick-a-pic-v2/pickapic/data/test-*.parquet",
            },
            cache_dir=cache_dir,
        )["test"]

        def preprocess(examples):
            images_batch = []
            ranks_batch = []
            for b0, b1, lab0 in zip(examples["jpg_0"], examples["jpg_1"], examples["label_0"]):
                imgs = [
                    Image.open(io.BytesIO(b0)).convert("RGB"),
                    Image.open(io.BytesIO(b1)).convert("RGB"),
                ]
                if lab0 == 0:
                    imgs = imgs[::-1]
                images_batch.append(imgs)
                ranks_batch.append([0, 1])
            return {
                "images": images_batch,
                "prompt": examples["caption"],
                "rank": ranks_batch,
            }
    elif 'imagerewarddb' in dataset_name.lower():
        train_data_dir = '/zjk_nas/zhiyi/data/ImageRewardDB'
        data_files = {"test": os.path.join(train_data_dir, "test.json")}
        dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)["test"]
        def preprocess(examples):
            # examples["image_path"]: list[list[str]]
            images_batch = [
                [Image.open(os.path.join(train_data_dir, p)).convert("RGB") for p in paths]
                for paths in examples["images"]
            ]
            if accelerator.is_main_process:
                print(examples["rank"])
            return {
                "images": images_batch,
                "prompt": examples["prompt"],
                "rank": examples["rank"],
            }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")

    # 用 map(batched=True) 应用（注意：会把 PIL 对象写进 dataset，不一定适合大数据/缓存）
    dataset = dataset.with_transform(preprocess)

    # -------------------------
    # dataloader + accelerate 分片
    # -------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=1, # must be 1
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    dataloader = accelerator.prepare(dataloader)  # 会自动给每个进程做 shard

    # -------------------------
    # eval
    # -------------------------
    local_sum = {name: 0.0 for name in scorers}
    local_cnt = 0

    iterator = dataloader
    if accelerator.is_main_process:
        iterator = tqdm(dataloader, desc="eval")

    for batch in iterator:
        # batch["images"]: list[list[PIL]]，长度为本进程的 batch_size
        for images, prompt, gt_rank in zip(batch["images"], batch["prompts"], batch["ranks"]):
            for name, scorer in scorers.items():
                scores = [scorer(prompt, img)[0] for img in images]
                print(scores)
                # print(gt_rank)
                pred_rank = scores_to_rankvec(scores, higher_is_better=True)
                local_sum[name] += inversion_score(pred_rank, gt_rank)
            local_cnt += 1

    # -------------------------
    # all-reduce 汇总
    # -------------------------
    # 把 dict 变成 tensor 做 gather/reduce
    names = list(scorers.keys())
    sum_tensor = torch.tensor([local_sum[n] for n in names], device=device, dtype=torch.float64)
    cnt_tensor = torch.tensor([local_cnt], device=device, dtype=torch.float64)

    sum_tensor = accelerator.reduce(sum_tensor, reduction="sum")
    cnt_tensor = accelerator.reduce(cnt_tensor, reduction="sum")

    if accelerator.is_main_process:
        acc = {n: (sum_tensor[i].item() / cnt_tensor.item()) for i, n in enumerate(names)}
        print(acc)


if __name__ == "__main__":
    main()
