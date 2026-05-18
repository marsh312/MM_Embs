from __future__ import annotations
import torch
from torch import nn
import logging
import math
import os
import requests
import base64
from io import BytesIO
from typing import List, Optional
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

# --- Helper Functions (Copied from source) ---
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
            headers = {'User-Agent': 'My User Agent 1.0'}
            image_obj = Image.open(requests.get(image, headers=headers, stream=True).raw)
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        else:
            image_obj = Image.open(image)
    
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, got {image}")
    
    image = image_obj.convert("RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(height, width, factor=size_factor)
    image = image.resize((resized_width, resized_height))
    return image

def custom_collate_fn(batch):
    return batch

# --- Main Class ---
class RzenEmbed(nn.Module):
    def __init__(
        self,
        model_name: str = "qihoo360/RzenEmbed",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_image_tokens=256,
        max_image_tokens=1280,
        max_length=2000,
        attn_implementation="flash_attention_2",
        processor: Optional[AutoProcessor] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        model_name = model_path or model_name
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        config.padding_side = "right"
        config.use_cache = False
        
        self.base = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, config=config,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        self.base.eval()
        self.normalize = True
        self.device = device
        self.base = self.base.to(self.device)
        
        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28
        self.max_length = max_length
        
        if processor is None:
            processor = AutoProcessor.from_pretrained(
                model_name, min_pixels=min_pixels, max_pixels=max_pixels
            )
        self.processor = processor
        self.processor.tokenizer.padding_side = 'right'
        self.defualt_instruction = 'You are a helpful assistant.'

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if inputs_embeds is None:
            # inputs_embeds = self.base.model.embed_tokens(input_ids)
            inputs_embeds = self.base.get_input_embeddings()(input_ids)
            has_image = (pixel_values is not None) and any([pv is not None for pv in pixel_values])
            
            if has_image:
                if isinstance(pixel_values, list):
                    pixel_values = torch.cat([torch.from_numpy(pv) for pv in pixel_values]).to(input_ids.device)
                    image_grid_thw = torch.cat([torch.from_numpy(thw) for thw in image_grid_thw]).to(input_ids.device)
                
                pixel_values = pixel_values.type(self.base.visual.get_dtype())
                image_embeds = self.base.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                image_mask = input_ids == self.base.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.base.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        
        # EOS Pooling
        pooling_mask = attention_mask if pooling_mask is None else pooling_mask
        sequence_lengths = pooling_mask.sum(dim=1) - 1
        batch_size = outputs.last_hidden_state.shape[0]
        embeddings = outputs.last_hidden_state[torch.arange(
            batch_size, device=outputs.last_hidden_state.device
        ), sequence_lengths]

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def _process_images(self, images):
        if isinstance(images, Image.Image) or isinstance(images, str):
            return [fetch_image(images)]
        return [fetch_image(i) for i in images]

    def embed(self, texts: list[str], images: list[Image.Image], **kwargs):
        # Determine batch size
        batch_size = len(texts) if texts is not None else len(images)
        input_texts, input_images = [], []
        
        # Corrected: System prompt is always fixed to default
        sys_instruction = self.defualt_instruction 
        
        for i in range(batch_size):
            text = texts[i] if texts is not None else None
            image = images[i] if images is not None else None
            input_str = ""
            processed_image = None
            
            if image is not None:
                processed_image = self._process_images(image)
                input_images += processed_image
                input_str += "<|vision_start|><|image_pad|><|vision_end|>" * len(processed_image)
            
            if text is not None:
                input_str += text
                
            # Corrected: Instruction from kwargs is ALREADY in input_str (via texts), sys prompt is fixed
            msg = f"<|im_start|>system\n{sys_instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
            input_texts.append(msg)
            
        if len(input_images) == 0:
            input_images = None

        inputs = self.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings = self.forward(**inputs)
        return embeddings

    def get_fused_embeddings(self, texts: list[str] = None, images: list = None, **kwargs):
        """
        Main interface for encoding.
        Correctly handles instruction prepending to texts.
        """
        assert texts or images, "Either 'texts' or 'images' must be provided - both cannot be None or empty"
        instruction = kwargs.pop('instruction', None)
        
        # Corrected: Prepend instruction to texts or create texts list if None
        if instruction is not None:
            if texts is not None:
                texts = [instruction + text for text in texts]
            elif images is not None:
                texts = [instruction] * (len(images) if not isinstance(images, DataLoader) else 0) # Handle DL below
                # Note: If passing DataLoader, we can't easily prepend text here.
                # However, your eval script passes lists, so this logic holds.
                # If images is DataLoader, we need custom logic inside the loop.
        
        # Setup DataLoader if needed
        if images is not None and not isinstance(images, DataLoader):
            batch_size = kwargs.pop('batch_size', 8)
            image_loader = DataLoader(
                images,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=8
            )
        elif isinstance(images, DataLoader):
            image_loader = images
            batch_size = image_loader.batch_size
        else:
            # Text only case
            batch_size = kwargs.pop('batch_size', 8)
            image_loader = None

        if texts is None:
            if image_loader is not None:
                n_batch = len(image_loader)
                # If using DataLoader and instruction provided but no texts, 
                # we need to generate texts for each batch inside the loop
                if instruction:
                    texts = [instruction] * (n_batch * batch_size) # Rough sizing, will slice in loop
            else:
                n_batch = 0
        else:
            n_batch = len(texts) // batch_size + int(len(texts) % batch_size > 0)
            if image_loader is None:
                image_loader = [None] * n_batch
        
        all_embeddings = []
        none_batch = [None] * batch_size
        
        # Iteration
        # Note: If images is DataLoader, tqdm iterates it. If List, we iterate range/loader manually.
        iterator = image_loader if image_loader else range(n_batch)
        
        for n, item in enumerate(tqdm(iterator, desc="Encoding", disable=False)):
            if image_loader:
                img_batch = item
            else:
                img_batch = None

            # Calculate text batch slice
            start_idx = n * batch_size
            end_idx = start_idx + batch_size
            
            # Handle text batching
            if texts is None:
                # If no texts list provided, check if we need to generate instruction-only text
                if instruction:
                    # Determine batch size for this iteration
                    curr_bs = len(img_batch) if img_batch else batch_size
                    text_batch = [instruction] * curr_bs
                else:
                    text_batch = none_batch[:len(img_batch)] if img_batch else None
            else:
                text_batch = texts[start_idx:end_idx]

            # Handle image batching if not from loader (e.g. text only mode)
            if img_batch is None and image_loader is None:
                img_batch = None
            
            # Skip empty batches
            if not text_batch and not img_batch: continue
            if text_batch and len(text_batch) == 0: continue

            # Align lengths
            if text_batch is not None and img_batch is not None:
                curr_batch_len = min(len(text_batch), len(img_batch))
                text_batch = text_batch[:curr_batch_len]
                img_batch = img_batch[:curr_batch_len]

            embeddings = self.embed(texts=text_batch, images=img_batch, **kwargs)
            all_embeddings.append(embeddings.cpu())
            
        # return torch.cat(all_embeddings, dim=0).numpy()
        return torch.cat(all_embeddings, dim=0).detach().cpu().to(torch.float32).numpy()