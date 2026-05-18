from __future__ import annotations

import logging
import os
from typing import cast, Dict, List, Optional, Union
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

class BGEM3TextEmbedding:
    def __init__(
        self,
        model_name: str = "/share/project/shared_models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        max_length: int = 8192,
        **kwargs,
    ) -> None:
        """
        Initialize BGE-M3 text embedding model
        
        Args:
            model_name: Model name or path
            device: Device to use for inference
            use_fp16: Whether to use fp16 for faster computation
            max_length: Maximum sequence length
        """
        self.model = BGEM3FlagModel(
            model_name, 
            use_fp16=use_fp16,
            **kwargs
        )
        self.device = device
        self.max_length = max_length
        self.normalize = True
        
        logger.info(f"Loaded BGE-M3 model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Max length: {max_length}")
        logger.info(f"Use FP16: {use_fp16}")

    def encode_queries(self, queries, batch_size: int = 256, max_length: int = None) -> np.ndarray:
        """
        Encode queries into dense embeddings
        
        Args:
            queries: Query dataset or text list
            batch_size: Batch size for encoding
            max_length: Maximum sequence length (if None, use model default)
            
        Returns:
            Dense embeddings as numpy array
        """
        if max_length is None:
            max_length = self.max_length
            
        # Extract text from queries
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
        
        logger.info(f"Encoding {len(input_texts)} queries with max_length={max_length}")
        
        # Use BGE-M3's encode method to get dense embeddings
        embeddings = self.model.encode(
            input_texts,
            batch_size=batch_size,
            max_length=max_length
        )['dense_vecs']
        
        return embeddings

    def encode_corpus(self, corpus, batch_size: int = 256, max_length: int = None) -> np.ndarray:
        """
        Encode corpus into dense embeddings
        
        Args:
            corpus: Corpus dataset or text list
            batch_size: Batch size for encoding
            max_length: Maximum sequence length (if None, use model default)
            
        Returns:
            Dense embeddings as numpy array
        """
        if max_length is None:
            max_length = self.max_length
            
        # Extract text from corpus
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
        
        logger.info(f"Encoding {len(input_texts)} corpus items with max_length={max_length}")
        
        # Use BGE-M3's encode method to get dense embeddings
        embeddings = self.model.encode(
            input_texts,
            batch_size=batch_size,
            max_length=max_length
        )['dense_vecs']
        
        return embeddings
