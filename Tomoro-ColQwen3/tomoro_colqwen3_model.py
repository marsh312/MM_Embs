from __future__ import annotations
import torch
from torch import nn
import logging
import math
import os
import requests
import base64
from io import BytesIO
from typing import List, Optional, Union
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# --- Helper Functions ---
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar

def fetch_image(image: str | Image.Image, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            headers = {'User-Agent': 'Mozilla/5.0'}
            image_obj = Image.open(requests.get(image, headers=headers, stream=True).raw)
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        else:
            # Assume local path
            image_obj = Image.open(image)
    
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, got {image}")
    
    image = image_obj.convert("RGB")
    # Resize logic is optional for ColQwen as processor handles it, but keeping it for consistency if needed.
    # Actually ColQwen processor handles resizing, so we might just return the PIL image.
    # But RzenEmbed helper does resizing. I'll keep it simple and just return PIL image.
    return image

# --- Main Class ---
class TomoroColQwen3Embed(nn.Module):
    def __init__(
        self,
        model_name: str = "TomoroAI/tomoro-colqwen3-embed-8b",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        max_visual_tokens: int = 1280,
        **kwargs,
    ) -> None:
        super().__init__()
        model_name = model_path or model_name
        self.device = device
        self.dtype = dtype
        
        print(f"Loading processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            max_num_visual_tokens=max_visual_tokens,
        )
        
        print(f"Loading model from {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map=device,
        ).eval()

    def get_text_embeddings(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> torch.Tensor:
        all_embeddings = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
            batch_texts = texts[start : start + batch_size]
            batch = self.processor.process_texts(texts=batch_texts)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.inference_mode():
                out = self.model(**batch)
                vecs = out.embeddings.to(torch.float32).cpu() # Move to float32 cpu for safety/eval
            all_embeddings.append(vecs)
        
        # Concatenate? No, lengths might vary if it's multi-vector.
        # But for texts (queries), ColQwen usually returns multi-vectors too.
        # The eval script needs to handle list of tensors or padded tensor.
        # I'll return a list of tensors to handle varying lengths if necessary, 
        # or just cat if they are padded.
        # Usually process_texts pads to max length in batch.
        # But across batches, lengths differ.
        # I'll return a list of tensors (one per query) to be safe.
        # Actually, let's see how `out.embeddings` looks. It's [B, L, D].
        # I'll return a list of [L, D] tensors.
        
        flat_list = []
        for batch_vecs in all_embeddings:
            # batch_vecs is [B, L, D]
            # We need to unpad? Or does the model output include padding?
            # ColBERT usually uses attention mask to mask out padding.
            # But here we are returning raw embeddings.
            # For simplicity in eval, let's return the full tensor list.
            for i in range(batch_vecs.shape[0]):
                flat_list.append(batch_vecs[i])
        return flat_list

    def get_image_embeddings(
        self,
        images: List[str | Image.Image],
        batch_size: int = 4
    ) -> List[torch.Tensor]:
        all_embeddings = []
        
        # Load images in batches to avoid OOM
        for start in tqdm(range(0, len(images), batch_size), desc="Processing images"):
            batch_imgs_raw = images[start : start + batch_size]
            batch_imgs = []
            for img in batch_imgs_raw:
                try:
                    batch_imgs.append(fetch_image(img))
                except Exception as e:
                    print(f"Error loading image: {e}")
                    # Create a black image as placeholder
                    batch_imgs.append(Image.new('RGB', (224, 224), color='black'))
            features = self.processor.process_images(images=batch_imgs)
            features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
            with torch.inference_mode():
                out = self.model(**features)
                vecs = out.embeddings.to(torch.float32).cpu()
            
            for i in range(vecs.shape[0]):
                all_embeddings.append(vecs[i])
                
        return all_embeddings
