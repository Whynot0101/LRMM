
import os
from datasets import load_dataset
from tqdm import tqdm
import json
train_data_dir = 'data/HPDv3'
data_files = {"test": os.path.join(train_data_dir, "test.json")}
dataset = load_dataset("json", data_files=data_files)["test"]
data = []
for example in tqdm(dataset):

    images = [os.path.join(train_data_dir, example["path1"]),  os.path.join(train_data_dir, example["path2"]),]
    prompt = example["prompt"]
    rank = [0, 1]
    data.append({
        'prompt': prompt,
        'images': images,
        'rank': rank,
    })


with open('data/HPDv3_test.json', 'w') as f:
    json.dump(data, f, indent=4)
