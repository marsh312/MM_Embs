# Requires vllm>=0.14.0
from io import BytesIO

import requests
import torch
from PIL import Image

from vllm import LLM


def get_image_from_url(url) -> Image.Image:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


model = LLM(model="/share/project/shared_models/Qwen3-VL-Embedding-2B", runner="pooling")

image = get_image_from_url("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
inputs = [
    {
        "prompt": "A woman playing with her dog on a beach at sunset.",
    },
    {
        "prompt": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
    },
    {
        "prompt": image_placeholder,
        "multi_modal_data": {"image": image},
    },
    {
        "prompt": f"{image_placeholder}\nA woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
        "multi_modal_data": {"image": image},
    },
]

outputs = model.embed(inputs)
embeddings = torch.tensor([o.outputs.embedding for o in outputs])
scores = embeddings[:2] @ embeddings[2:].T
print(scores.tolist())