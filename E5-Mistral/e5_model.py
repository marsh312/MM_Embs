from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class E5MistralModel:
    """E5-Mistral-7b-instruct embedding model wrapper"""
    
    def __init__(
        self,
        model_name: str = "/share/project/shared_models/models--intfloat--e5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        max_length: int = 4096,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.max_length = max_length
        
        logger.info(f"Initializing E5MistralModel")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        if "attn_implementation" in kwargs:
            logger.info(f"Attention implementation: {kwargs['attn_implementation']}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
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
            
            self.model = self.model.to(device)
            self.model.eval()
            
            logger.info(f"E5-Mistral model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load E5-Mistral model from {model_name}")
            raise RuntimeError(f"Failed to initialize E5-Mistral model: {e}") from e

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.no_grad()
    def _encode_batch(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode a list of texts.
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the input texts
            batch_dict = self.tokenizer(
                batch_texts, 
                max_length=self.max_length, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            
            batch_dict = batch_dict.to(self.device)
            
            outputs = self.model(**batch_dict)
            embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])

    def encode_queries(self, queries: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode queries with instruction.
        Instruction: Given a web search query, retrieve relevant passages that answer the query
        Format: Instruct: {task_description}\nQuery: {query}
        """
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        formatted_queries = [f'Instruct: {task}\nQuery: {q}' for q in queries]
        return self._encode_batch(formatted_queries, batch_size=batch_size)

    def encode_corpus(self, corpus: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode corpus without instruction.
        """
        return self._encode_batch(corpus, batch_size=batch_size)
