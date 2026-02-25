# import
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image
import torch
import sys
from src.mps.trainer.models.clip_model import CLIPModel
import src.mps.trainer as trainer
sys.modules["trainer"] = trainer  

class MPS:
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.image_processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)
        model_ckpt_path = "rewards/MPS_overall_checkpoint.pth"
        self.model = CLIPModel("openai/clip-vit-base-patch32")
        self.model=torch.load(model_ckpt_path, weights_only=False)
        # self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    def __call__(self, prompt, image):
        condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things." 
        def _process_image(image):
            if isinstance(image, dict):
                image = image["bytes"]
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            if isinstance(image, str):
                image = Image.open( image )
            image = image.convert("RGB")
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
            return pixel_values
        
        def _tokenize(caption):
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            return input_ids
        
        image_inputs = _process_image(image).to(self.device)
        text_inputs = _tokenize(prompt).to(self.device)
        condition_inputs = _tokenize(condition).to(self.device)

        with torch.no_grad():
            text_features, image_0_features, image_1_features = self.model(text_inputs, image_inputs, condition_inputs)
            image_0_features = image_0_features / image_0_features.norm(dim=-1, keepdim=True)
            image_1_features = image_1_features / image_1_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_scores = self.model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_0_features))
            score = image_0_scores[0]

        return score
