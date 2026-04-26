
import os
from datasets import load_dataset
from tqdm import tqdm
import json
train_data_dir = 'data/HPDv2'
data_files = {"test": os.path.join(train_data_dir, "test.json")}
dataset = load_dataset("json", data_files=data_files)["test"]
data = []
for example in tqdm(dataset):

    images = [os.path.join(train_data_dir,"test", path) for path in example["image_path"]]
    prompt = example["prompt"]
    rank = example["rank"]
    data.append({
        'prompt': prompt,
        'images': images,
        'rank': rank,
    })


with open('data/HPDv2_test.json', 'w') as f:
    json.dump(data, f, indent=4)
