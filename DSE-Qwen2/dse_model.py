import torch
import requests
from PIL import Image
from io import BytesIO
from typing import List, Optional, Union
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import time
import os

def fetch_image(image: str | Image.Image) -> Image.Image:
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
                import base64
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        else:
            # Assume local path
            image_obj = Image.open(image)
    
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, got {image}")
    
    return image_obj.convert("RGB")

class DSEQwenEmbed:
    def __init__(
        self,
        model_name: str = "MrLight/dse-qwen2-2b-mrl-v1",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        dimension: int = 1536
    ) -> None:
        model_name = model_path or model_name
        self.device = device
        self.dtype = dtype
        self.dimension = dimension
        
        print(f"Loading model and processor from {model_name}...")
        min_pixels = 1*28*28
        max_pixels = 2560*28*28
        
        t0 = time.time()
        print("[DSE] begin load processor")
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels,
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        print(f"[DSE] load processor cost {time.time() - t0:.4f} s")

        print("[DSE] begin from_pretrained", flush=True)
        t1 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            # use_safetensors=False, 
            # trust_remote_code=True,
            # local_files_only=True,
            # low_cpu_mem_usage=True,
        )
        print(f"[DSE] from_pretrained done: {time.time()-t1:.3f}s", flush=True)

        print("[DSE] begin to(device)", flush=True)
        t2 = time.time()
        self.model = self.model.to(device).eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[DSE] to(device) done: {time.time()-t2:.3f}s", flush=True)
                
        # Configure padding
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"

    def _get_embedding(self, last_hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps[:, :dimension], p=2, dim=-1)
        return reps

    def get_query_embeddings(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> List[torch.Tensor]:
        all_embeddings = []
        
        for start in tqdm(range(0, len(texts), batch_size), desc="Processing queries"):
            batch_texts = texts[start : start + batch_size]
            
            # Prepare messages
            messages = []
            for text in batch_texts:
                msg = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': Image.new('RGB', (28, 28)), 'resized_height': 1, 'resized_width': 1}, # Dummy image
                            {'type': 'text', 'text': f'Query: {text}'},
                        ]
                    }
                ]
                messages.append(msg)
            
            # Process inputs
            text_inputs = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=text_inputs, 
                images=image_inputs, 
                videos=video_inputs, 
                padding='longest', 
                return_tensors='pt'
            ).to(self.device)
            
            # Forward pass
            # We need cache_position for generation-like models usually, but for embedding extraction?
            # The example code uses prepare_inputs_for_generation.
            # Let's follow the example exactly.
            
            # Wait, cache_position is needed if we use prepare_inputs_for_generation
            # The example code:
            # cache_position = torch.arange(0, len(query_texts)) -- Wait, len(query_texts) is batch size?
            # No, usually cache_position is sequence length related?
            # Actually, prepare_inputs_for_generation documentation says cache_position is for static cache.
            # But here we are just doing one forward pass.
            # Let's check the example again carefully.
            # cache_position = torch.arange(0, len(query_texts)) -> This looks wrong if it's batch size.
            # It should be sequence length?
            # "cache_position = torch.arange(0, len(query_texts))" -> In the example, len(query_texts) is 2.
            # But usually prepare_inputs_for_generation needs seq_len if it's for generation.
            # However, Qwen2VL might be different.
            # Let's look at `inputs.input_ids.shape`. It is [B, L].
            # If we don't pass cache_position, does it work?
            # The example code:
            # cache_position = torch.arange(0, len(query_texts))
            # query_inputs = model.prepare_inputs_for_generation(**query_inputs, cache_position=cache_position, use_cache=False)
            # 
            # If len(query_texts) is batch size, then cache_position is [0, 1].
            # But input_ids is [2, L].
            # This seems suspicious. Maybe it meant `inputs.input_ids.shape[1]`?
            # Or maybe `cache_position` is not needed if we just call model()?
            # `model(**query_inputs)` usually works for HF models.
            # But Qwen2VL might require it.
            
            # Let's try WITHOUT prepare_inputs_for_generation first, as it's cleaner.
            # If that fails, we can check.
            # Actually, the example uses `model.prepare_inputs_for_generation`.
            # And `cache_position` logic in example: `torch.arange(0, len(query_texts))`
            # If query_texts has 2 items, it's `[0, 1]`.
            # This is extremely short for cache_position.
            # I suspect the example code on HF might have a quirk or I'm misunderstanding.
            # Ah, `process_vision_info` returns `image_inputs` which are list of images.
            
            # Let's stick to standard `model(**inputs)`.
            # If `prepare_inputs_for_generation` is strictly required for Qwen2VL to handle images correctly (e.g. patching), then we must use it.
            # But usually `processor` output is ready for `model`.
            # The example explicitly uses it.
            
            # Let's try to assume `model(**inputs)` works.
            # If not, I'll revisit.
            
            with torch.no_grad():
                output = self.model(**inputs, return_dict=True, output_hidden_states=True)
                
            embeddings = self._get_embedding(output.hidden_states[-1], self.dimension)
            vecs = embeddings.to(torch.float32).cpu()
            
            for i in range(vecs.shape[0]):
                all_embeddings.append(vecs[i])
        
        return all_embeddings

    def get_image_embeddings(
        self,
        images: List[str | Image.Image],
        batch_size: int = 4
    ) -> List[torch.Tensor]:
        all_embeddings = []
        
        for start in tqdm(range(0, len(images), batch_size), desc="Processing images"):
            batch_imgs_raw = images[start : start + batch_size]
            batch_imgs = []
            for img in batch_imgs_raw:
                try:
                    batch_imgs.append(fetch_image(img))
                except Exception as e:
                    print(f"Error loading image: {e}")
                    batch_imgs.append(Image.new('RGB', (224, 224), color='white'))
            
            # Prepare messages
            messages = []
            for img in batch_imgs:
                msg = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': img}, # resized_height/width optional? Example says "adjust for efficiency".
                            # Let's not force resize unless necessary.
                            {'type': 'text', 'text': 'What is shown in this image?'}
                        ]
                    }
                ]
                messages.append(msg)
                
            # Process inputs
            text_inputs = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=text_inputs, 
                images=image_inputs, 
                videos=video_inputs, 
                padding='longest', 
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model(**inputs, return_dict=True, output_hidden_states=True)
                
            embeddings = self._get_embedding(output.hidden_states[-1], self.dimension)
            vecs = embeddings.to(torch.float32).cpu()
            
            for i in range(vecs.shape[0]):
                all_embeddings.append(vecs[i])
                
        return all_embeddings
