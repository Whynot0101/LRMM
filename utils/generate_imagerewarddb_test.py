from datasets import load_dataset
from datasets.features import image
import json
import os
dataset = load_dataset(
            "parquet",
            data_files={
                "test": "data/ImageRewardDB/metadata-test.parquet",
            },
            cache_dir=None,
        )["test"]

test_dict = {}

for i, example in enumerate(dataset):
    if example["prompt_id"] not in test_dict:
        test_dict[example["prompt_id"]] =  {"prompt": example["prompt"], "images": [os.path.join('data', 'ImageRewardDB', example["image_path"])], "rank": [int(example["rank"])-1]}
    else:
        assert os.path.join('data', 'ImageRewardDB', example["image_path"]) not in test_dict[example["prompt_id"]]["images"]
        test_dict[example["prompt_id"]]["images"].append(os.path.join('data', 'ImageRewardDB', example["image_path"]))
        test_dict[example["prompt_id"]]["rank"].append(int(example["rank"])-1)

test_json = [i for i in test_dict.values()]
# print(test_json)

with open('data/ImageRewardDB_test.json', 'w') as f:
    json.dump(test_json, f)
