import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def process_item(data, prefix_parts):
    try:
        # 1. 构造完整路径
        p1 = os.path.join(*prefix_parts, data["path1"])
        p2 = os.path.join(*prefix_parts, data["path2"])
        
        # 2. 检查物理文件是否存在
        if not (os.path.exists(p1) and os.path.exists(p2)):
            print(f"{p1} or {p2} donest exist")
            return None
        
        # 3. 检查图片是否损坏 (Integrity Check)
        # verify() 仅读取文件头检查完整性，不解码像素，速度非常快
        with Image.open(p1) as img1:
            img1.verify()
        with Image.open(p2) as img2:
            img2.verify()
        
        # 4. 如果没问题，更新路径并返回
        data["path1"] = p1
        data["path2"] = p2
        data.pop('confidence')
        data.pop('choice_dist')
        return data
    except Exception:
        # 捕获包括 FileNotFoundError, UnidentifiedImageError 等所有异常
        return None

def thread_process_list(data_list, prefix_parts, desc):
    processed_results = []
    failed_count = 0
    with ThreadPoolExecutor(max_workers=80) as executor:
        # 提交任务
        futures = [executor.submit(process_item, item, prefix_parts) for item in data_list]
        
        for future in tqdm(futures, desc=desc):
            result = future.result()
            if result is not None:
                processed_results.append(result)
            else:
                failed_count += 1
    return processed_results, failed_count

# --- 执行清洗逻辑 ---

with open('HPDv3/train.json') as f:
    hpdv3_raw = json.load(f)
hpdv3_train, hpdv3_failed = thread_process_list(hpdv3_raw, ("HPDv3",), "Checking HPDv3")

with open('HPDv3/pickapic.json') as f:
    pickapic_raw = json.load(f)
pickapic_train, pickapic_failed = thread_process_list(pickapic_raw, ("Pickapic","images"), "Checking Pickapic")

with open('HPDv3/imagereward.json') as f:
    imagereward_raw = json.load(f)
imagereward_train, imagereward_failed = thread_process_list(imagereward_raw, ("",), "Checking ImageReward")

# 合并所有清洗后的有效数据
train_data = hpdv3_train + pickapic_train + imagereward_train


print("\n" + "="*35)
print(f"HPDv3: {len(hpdv3_train)} valid, {hpdv3_failed} deleted")
print(f"Pickapic: {len(pickapic_train)} valid, {pickapic_failed} deleted")
print(f"ImageReward: {len(imagereward_train)} valid, {imagereward_failed} deleted")
print("-" * 35)
print(f"Total valid items: {len(train_data)}")
print(f"Total failed/deleted: {hpdv3_failed + pickapic_failed + imagereward_failed}")
print("="*35)

print(f"Saving to train_all.json...")
with open('train_all.json', 'w') as f:
    json.dump(train_data, f, indent=4)

print("Done!")
