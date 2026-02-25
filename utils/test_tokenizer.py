from diffusers import QwenImagePipeline
from transformers import Qwen2Tokenizer

tokenizer = Qwen2Tokenizer.from_pretrained(
        "/mnt/ramdisk/qwen",
        subfolder="tokenizer"
    )

special_tokens = {"additional_special_tokens": ["<reward>"]
        }
num_added = tokenizer.add_special_tokens(special_tokens)
print("新增 token 数量：", num_added)

ids = tokenizer(
    "<reward> A painting of a squirrel eating a burger",
    return_tensors="pt",
).input_ids[0]
print(ids)
tokens = tokenizer.convert_ids_to_tokens(ids)
print(tokens)

