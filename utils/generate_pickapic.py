import os
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# --- 配置 ---
output_base_dir = "data/Pickapic"
images_dir = os.path.join(output_base_dir, "images")
os.makedirs(images_dir, exist_ok=True)

def save_original_format_task(img_bytes, name):
    """保存单张图片任务"""
    try:
        path = os.path.join(images_dir, f"{name}.jpg")
        # 如果图片已存在则跳过，Pickapic 有大量重复图片，这样能极大提速
        if os.path.exists(path):
            return True
            
        # 如果是字节流则打开，如果是 PIL 对象则跳过 BytesIO
        if isinstance(img_bytes, bytes):
            img = Image.open(io.BytesIO(img_bytes))
        else:
            img = img_bytes
            
        img = img.convert('RGB')
        img.save(path, format="JPEG", quality=100, subsampling=0)
        return True
    except Exception as e:
        print(f"Error saving {name}: {e}")
        return False

# 1. 加载并过滤数据集
dataset = load_dataset(
    "parquet",
    data_files={
        "train": "data/Pickapic/data/train-*.parquet",
        "test": "data/Pickapic/data/test-*.parquet"
    },
)

for split in ['train', 'test']:
    orig_len = dataset[split].num_rows
    # 过滤 label_0 不在 (0, 1) 之间的（即过滤掉 0.5 或者无效数据）
    dataset[split] = dataset[split].filter(lambda x: x['label_0'] in (0, 1), num_proc=4)
    new_len = dataset[split].num_rows
    print(f"{split}: Eliminated {orig_len - new_len}/{orig_len} entries.")

# 2. 多线程处理逻辑
def process_split(split_name):
    metadata = []
    futures = []
    
    # 使用 ThreadPoolExecutor
    # 如果图片非常多，建议分批 submit，否则 futures 列表会撑爆内存
    with ThreadPoolExecutor(max_workers=16) as executor:
        print(f"正在提交 {split_name} 任务并处理元数据...")
        
        # 进度条 1: 提交任务和整理元数据的进度
        for example in tqdm(dataset[split_name], desc=f"Submitting {split_name}"):
            is_zero_better = int(example['label_0']) == 1 # 注意：Pickapic 中 1 通常代表 image_0 更好
            
            # 统一命名和路径逻辑
            uid_0 = example["image_0_uid"]
            uid_1 = example["image_1_uid"]
            
            # 提交保存任务 (注意这里去掉了之前代码里的中括号，直接传 UID 字符串)
            # futures.append(executor.submit(save_original_format_task, example["jpg_0"], uid_0))
            # futures.append(executor.submit(save_original_format_task, example["jpg_1"], uid_1))
            
            # 整理元数据 (path1 永远放更好的图)
            win_uid = uid_0 if is_zero_better else uid_1
            lose_uid = uid_1 if is_zero_better else uid_0
            
            if split_name=='train':
                metadata.append({
                    'prompt': example['caption'],
                    'path1': f"data/Pickapic/images/{win_uid}.jpg",
                    'path2': f"data/Pickapic/images/{lose_uid}.jpg",
                    "confidence": [],
                    "choice_dist": [],
                })
            elif split_name=='test':
                metadata.append({
                    "prompt": example['caption'],
                    "images": [f"data/Pickapic/images/{win_uid}.jpg", f"data/Pickapic/images/{lose_uid}.jpg"],
                    "rank": [0, 1],
                })

        # 进度条 2: 等待所有图片保存完成的进度
        # print(f"等待 {split_name} 图片写入硬盘...")
        # for future in tqdm(as_completed(futures), total=len(futures), desc=f"Writing {split_name}"):
        #     future.result() # 获取结果，如果有报错会在这里抛出

    return metadata

# 执行处理
train_metadata = process_split('train')
test_metadata = process_split('test')

# 3. 保存 JSON
with open("data/Pickapic_train.json", 'w', encoding='utf-8') as f:
    json.dump(train_metadata, f, indent=4, ensure_ascii=False)

with open("data/Pickapic_test.json", 'w', encoding='utf-8') as f:
    json.dump(test_metadata, f, indent=4, ensure_ascii=False)

print("全部完成！")
