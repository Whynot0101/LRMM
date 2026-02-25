import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = "/zjk_nas/zhiyi/data/HPDv3"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(BASE_DIR, "train.json"),
        help="原始 json 文件路径"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(BASE_DIR, "train_filtered.json"),
        help="过滤后的 json 文件路径"
    )
    parser.add_argument(
        "--path1_key", type=str, default="path1",
        help="第一张图片路径的字段名"
    )
    parser.add_argument(
        "--path2_key", type=str, default="path2",
        help="第二张图片路径的字段名"
    )
    parser.add_argument(
        "--num_workers", type=int, default=32,
        help="检查文件存在性的线程数"
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    def check_example(idx_ex):
        idx, ex = idx_ex
        p1_rel = ex.get(args.path1_key)
        p2_rel = ex.get(args.path2_key)

        p1 = os.path.join(BASE_DIR, p1_rel) if p1_rel else ""
        p2 = os.path.join(BASE_DIR, p2_rel) if p2_rel else ""

        ok = (
            isinstance(p1, str) and p1 != "" and os.path.exists(p1)
            and isinstance(p2, str) and p2 != "" and os.path.exists(p2)
        )
        return idx, ok  # 返回索引方便后面重建列表

    kept_flags = [False] * len(data)
    dropped = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex_pool:
        futures = [
            ex_pool.submit(check_example, (i, ex))
            for i, ex in enumerate(data)
        ]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx, ok = fut.result()
            kept_flags[idx] = ok
            if not ok:
                dropped += 1

    kept = [ex for ex, keep in zip(data, kept_flags) if keep]

    print(f"Total: {len(data)}, kept: {len(kept)}, dropped: {dropped}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
