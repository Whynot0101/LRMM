from numpy import dtype
import torch
import clip
from PIL import Image

class HPS:
    def __init__(self, device="cuda", dtype=torch.float32):
        self.model, self.preprocess = clip.load("ViT-L/14", device="cuda")
        params = torch.load("path/to/hpc.pth")['state_dict']
        self.model.load_state_dict(params)
        self.model.to(device, dtype=dtype)
        self.model.eval()
        self.device = device
        self.dtype = dtype

    def __call__(self, prompt, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        text = clip.tokenize(["your prompt here"]).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            score = image_features @ text_features.T

            return score[0]