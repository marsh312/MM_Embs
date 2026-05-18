from __future__ import annotations

import logging
import os
from typing import cast, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class Qwen3EmbeddingModel:
    """Qwen3-Embedding-8B text embedding model wrapper"""
    
    def __init__(
        self,
        model_name: str = "/share/project/shared_models/Qwen3-Embedding-8B/",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        max_length: int = 8192,
        **kwargs,
    ) -> None:
        """
        Initialize Qwen3 embedding model
        
        Args:
            model_name: Model name or path
            device: Device to use for inference
            use_fp16: Whether to use fp16 for faster computation
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.max_length = max_length
        
        logger.info(f"Initializing Qwen3EmbeddingModel")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Max length: {max_length}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            
            # Load model with appropriate dtype and device settings
            if use_fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
                
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **kwargs
            )
            
            # Move model to device and set to eval mode
            self.model = self.model.to(device)
            self.model.eval()
            
            logger.info(f"Qwen3 embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3 embedding model from {model_name}")
            raise RuntimeError(f"Failed to initialize Qwen3 embedding model: {e}") from e

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last token pooling as used in Qwen3-Embedding"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.no_grad()
    def _encode_batch(self, texts: List[str], batch_size: int = 256, max_length: int = None) -> np.ndarray:
        """Internal method to encode a batch of texts"""
        if max_length is None:
            max_length = self.max_length
            
        all_embeddings = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = batch_dict.to(self.device)
            
            # Get model outputs
            outputs = self.model(**batch_dict)
            
            # Apply last token pooling
            embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy and move to CPU
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        if all_embeddings:
            return np.concatenate(all_embeddings, axis=0)
        else:
            return np.array([])
            
    def _extract_text_from_queries(self, queries):
        """Helper method to extract text from queries in various formats"""
        if hasattr(queries, '__getitem__') and hasattr(queries, '__len__'):
            # Dataset object
            input_texts = []
            for i in range(len(queries)):
                query_item = queries[i]
                if isinstance(query_item, dict):
                    text = query_item.get("q_text", "")
                else:
                    text = str(query_item)
                input_texts.append(text)
        elif isinstance(queries, dict) and "q_text" in queries:
            # Dictionary format
            input_texts = queries["q_text"]
        elif isinstance(queries, list):
            # Direct text list
            input_texts = queries
        else:
            raise ValueError(f"Unsupported queries format: {type(queries)}")
        
        return input_texts
    
    def _extract_text_from_corpus(self, corpus):
        """Helper method to extract text from corpus in various formats"""
        if hasattr(corpus, '__getitem__') and hasattr(corpus, '__len__'):
            # Dataset object
            input_texts = []
            for i in range(len(corpus)):
                corpus_item = corpus[i]
                if isinstance(corpus_item, dict):
                    text = corpus_item.get("text", "")
                else:
                    text = str(corpus_item)
                input_texts.append(text)
        elif isinstance(corpus, dict) and "text" in corpus:
            # Dictionary format
            input_texts = corpus["text"]
        elif isinstance(corpus, list):
            # Direct text list
            input_texts = corpus
        else:
            raise ValueError(f"Unsupported corpus format: {type(corpus)}")
        
        return input_texts

    def encode_queries(self, queries, batch_size: int = 256, max_length: int = None) -> np.ndarray:
        """Encode queries into dense embeddings"""
        input_texts = self._extract_text_from_queries(queries)
        
        logger.info(f"Encoding {len(input_texts)} queries with max_length={max_length or self.max_length}")
        
        return self._encode_batch(input_texts, batch_size, max_length)

    def encode_corpus(self, corpus, batch_size: int = 256, max_length: int = None) -> np.ndarray:
        """Encode corpus into dense embeddings"""
        input_texts = self._extract_text_from_corpus(corpus)
        
        logger.info(f"Encoding {len(input_texts)} corpus items with max_length={max_length or self.max_length}")
        
        return self._encode_batch(input_texts, batch_size, max_length)
