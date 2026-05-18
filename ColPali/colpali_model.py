import torch
import requests
from PIL import Image
from io import BytesIO
from typing import List, Optional, Union
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor

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

class ColPaliEmbed:
    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        model_name = model_path or model_name
        self.device = device
        self.dtype = dtype
        
        print(f"Loading model and processor from {model_name}...")
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def get_query_embeddings(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> List[torch.Tensor]:
        all_embeddings = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Processing queries"):
            batch_texts = texts[start : start + batch_size]
            batch_queries = self.processor.process_queries(batch_texts).to(self.device)
            
            with torch.inference_mode():
                embeddings = self.model(**batch_queries)
                # embeddings is [B, L, D] (or similar, depending on output)
                # Move to cpu float32
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
                    # Create a white image as placeholder (as per example)
                    batch_imgs.append(Image.new('RGB', (224, 224), color='white'))
            
            batch_images = self.processor.process_images(batch_imgs).to(self.device)
            
            with torch.inference_mode():
                embeddings = self.model(**batch_images)
                vecs = embeddings.to(torch.float32).cpu()
            
            for i in range(vecs.shape[0]):
                all_embeddings.append(vecs[i])
                
        return all_embeddings
