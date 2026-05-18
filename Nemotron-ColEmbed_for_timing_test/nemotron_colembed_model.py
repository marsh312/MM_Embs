import torch
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from transformers import AutoModel
from transformers.image_utils import load_image
from PIL import Image
import time
from tqdm import tqdm

def _pad_and_stack_embeddings(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad embedding tensors to uniform sequence length and concatenate.
    Args:
        tensors: List of tensors with shape (batch, seq_len, hidden_dim).
            Each tensor may have a different seq_len.
    Returns:
        Concatenated tensor with shape (total_batch, max_seq_len, hidden_dim),
        where sequences shorter than max_seq_len are zero-padded.
    """
    if not tensors:
        raise ValueError("Cannot pad empty tensor list")

    max_seq_len = max(t.shape[1] for t in tensors)
    total_docs = sum(t.shape[0] for t in tensors)
    hidden_dim = tensors[0].shape[2]
    dtype = tensors[0].dtype

    # Pre-allocate result tensor
    result = torch.zeros(total_docs, max_seq_len, hidden_dim, dtype=dtype)

    # Copy in-place and release references to free memory
    offset = 0
    for i in range(len(tensors)):
        t = tensors[i]
        tensors[i] = None  # Release reference immediately
        batch_size = t.shape[0]
        seq_len = t.shape[1]
        result[offset : offset + batch_size, :seq_len, :] = t
        offset += batch_size
        del t

    return result

class NemotronColEmbed:
    def __init__(
        self,
        model_path: str,
        device_map: Union[str, dict] = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        ).eval()


    def forward_queries(self, queries: List[str], batch_size: int = 32) -> torch.Tensor:
        embeddings = self.model.forward_queries(queries, batch_size=batch_size)
        return embeddings

    def _load_image_safe(self, src: Union[str, Image.Image]) -> Image.Image:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        try:
            img = load_image(src)
            return img.convert("RGB")
        except Exception:
            data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAYAAABu4qVbAAAAAklEQVR4AewaftIAAAgwAeyfQy4AAAAA"
            print(f"Warning: Failed to load image {src}, using placeholder instead.")
            if isinstance(src, str) and src.startswith("data:image"):
                return load_image(src).convert("RGB")
            return load_image(data_url).convert("RGB")

    def forward_images(
        self,
        images_or_paths: List[Union[str, Image.Image]],
        batch_size: int = 32,
        num_workers: int = 16,
    ) -> torch.Tensor:
        batches = []
        total_images = len(images_or_paths)
        for batch_idx, i in enumerate(tqdm(range(0, total_images, batch_size), desc="Encoding images")):
            chunk = images_or_paths[i : i + batch_size]
            loaded = [self._load_image_safe(x) for x in chunk]
            emb = self.model.forward_images(loaded, batch_size=batch_size) # (batch_size, seq_len, hidden_dim), tensor on cpu
            batches.append(emb)
        print("Finished encoding images loop. Stacking embeddings...", flush=True)
        return _pad_and_stack_embeddings(batches) # (total_images, max_seq_len, hidden_dim); tensor on cpu

    def get_scores(self, query_embeddings: Union[torch.Tensor, List[torch.Tensor]], passage_embeddings: Union[torch.Tensor, List[torch.Tensor]], batch_size: Optional[int] = 128) -> torch.Tensor:
        # return self.model.get_scores(query_embeddings, passage_embeddings, batch_size=batch_size)
        return self.model.get_scores_fast(query_embeddings, passage_embeddings, query_batch_size=batch_size, passage_batch_size=batch_size)
