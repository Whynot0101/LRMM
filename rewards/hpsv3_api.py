import base64
import requests
import json
import os
import torch
import io 
from PIL import Image

def bytes_to_data_url(b, mime="image/jpeg"):
    return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

class HPSv3Scorer():
    def __init__(self, device="cuda", dtype=torch.float32):
        self.api_url = "http://33.200.227.184:8000/infer_score"

    def __call__(self, prompt, image):

        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            image = bytes_to_data_url(buf.getvalue(), mime=f"image/{'jpeg'}")

        payload = {
        "image_path": image,
        "prompt": prompt,
        }
        # 3. 发送 POST 请求
        for _ in range(20):
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                score = response.json()["score"]
                return [score]
            else:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                print(f"错误详情: {response.json()}")
                continue

        return None

if __name__ == "__main__":
    scorer = HPSv3Scorer(
        device="cuda",
        dtype=torch.float32
    )
    score = scorer("Sunset reflecting on a crygangster cat wearing snapback and golden chain on neck with dollar sign pendantstal ball", "/nas/zhiyi/data/student_600.jpg")
    print(score)