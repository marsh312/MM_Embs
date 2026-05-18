# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.
"""Qwen3VLNemotronEmbed: Vision-Language Embedding Model with ColBERT-style scoring.

This module provides a bidirectional vision-language model for document retrieval
and embedding tasks, based on the Qwen3VL architecture with bidirectional attention.
"""

from contextlib import nullcontext
from typing import Dict, List, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithPast,
    Cache,
    FlashAttentionKwargs,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Unpack,
    auto_docstring,
    check_model_inputs,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
import os
import time


TV = TypeVar("TV")


class ListDataset(TorchDataset[TV]):
    """Simple dataset wrapper for list elements."""

    def __init__(self, elements: List[TV]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> TV:
        return self.elements[idx]


class Qwen3VLNemotronEmbedConfig(Qwen3VLConfig):
    """Configuration for Qwen3VLNemotronEmbed models."""

    model_type = "qwen3_vl_nemotron_embed"

    pooling: str

    def __init__(
        self,
        pooling: str = "colbert",
        **kwargs,
    ):
        self.pooling = pooling
        super().__init__(**kwargs)


def _create_bidirectional_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """Create bidirectional attention mask based on attention implementation."""
    if config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None
    elif config._attn_implementation == "eager":
        if attention_mask is not None:
            return _prepare_4d_attention_mask(
                attention_mask,
                dtype=input_embeds.dtype,
                tgt_len=input_embeds.shape[1],
            )
        return None
    else:
        if attention_mask is not None:
            return _prepare_4d_attention_mask(
                attention_mask,
                dtype=input_embeds.dtype,
                tgt_len=input_embeds.shape[1],
            )
        return None


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


class Qwen3VLNemotronEmbedTextModel(Qwen3VLTextModel):
    """Bidirectional text model for Qwen3VLNemotronEmbed."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """
        visual_pos_masks (`torch.Tensor`, *optional*):
            Boolean mask indicating positions of visual tokens in the sequence.
            Used for deepstack processing to identify where to inject visual features.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            List of visual embeddings from intermediate vision encoder layers.
            Each tensor corresponds to a decoder layer where visual features are injected.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = _create_bidirectional_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            if deepstack_visual_embeds is not None and layer_idx in range(
                len(deepstack_visual_embeds)
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


class Qwen3VLNemotronEmbedVisionLanguageModel(Qwen3VLModel):
    """Vision-language model with bidirectional text attention."""

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLNemotronEmbedTextModel._from_config(
            config.text_config
        )
        self.rope_deltas = None

        self.post_init()


class Qwen3VLNemotronEmbedForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3VLNemotronEmbed for conditional generation (base class)."""

    config: Qwen3VLNemotronEmbedConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLNemotronEmbedVisionLanguageModel._from_config(config)


class EmbeddingMixin:
    """Mixin providing high-level embedding extraction methods."""

    def _get_processor(self):
        """Get or create the processor for this model."""
        if not hasattr(self, "_processor") or self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
        return self._processor
    
    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000.0

    @staticmethod
    def _try_get_rss_gb() -> float:
        try:
            import psutil
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / (1024 ** 3)
        except Exception:
            return float("nan")

    @staticmethod
    def _cuda_mem_mb(device: torch.device) -> tuple[float, float]:
        if device.type != "cuda":
            return (0.0, 0.0)
        return (
            torch.cuda.memory_allocated(device) / (1024 ** 2),
            torch.cuda.memory_reserved(device) / (1024 ** 2),
        )

    def process_queries(self, queries: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """Process text queries for embedding extraction."""
        return self._get_processor().process_queries(queries, **kwargs)

    def process_documents(
        self, documents: Union[Dict, List[Dict]], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Process documents (image + text) for embedding extraction."""
        return self._get_processor().process_documents(documents, **kwargs)

    def _extract_embeddings(
        self, dataloader: DataLoader, is_query: bool
    ) -> torch.Tensor:
        """Extract embeddings from a dataloader.

        Args:
            dataloader: DataLoader yielding batches of processed inputs.
            is_query: Whether these are query embeddings (for progress message).

        Returns:
            Tensor of embeddings with shape (num_samples, max_seq_len, hidden_dim).
        """
        device = next(self.parameters()).device
        embedding_batches = []
        message = "query" if is_query else "document"

        it = iter(dataloader)
        step = 0

        pbar = tqdm(desc=f"Extracting {message} embeddings...")

        while True:
            # 1) 等 dataloader 出 batch 的时间
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()

            # 2) H2D: to(device) 时间
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()

            with torch.inference_mode():
                with (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                ):
                    if "pixel_values" in batch and batch["pixel_values"] is None:
                        batch.pop("pixel_values")

                    batch = {k: v.to(device) for k, v in batch.items()}

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t3 = time.perf_counter()

                    # 3) forward 时间，用 CUDA Event 计时
                    fwd_ms = float("nan")
                    if device.type == "cuda":
                        start_evt = torch.cuda.Event(enable_timing=True)
                        end_evt = torch.cuda.Event(enable_timing=True)
                        start_evt.record()

                    embeddings = self(**batch, output_hidden_states=True).hidden_states[-1]
                    embeddings = embeddings * batch["attention_mask"].unsqueeze(-1)
                    embeddings = F.normalize(embeddings, dim=-1)

                    if device.type == "cuda":
                        end_evt.record()
                        torch.cuda.synchronize(device)
                        fwd_ms = start_evt.elapsed_time(end_evt)

            if not torch.isfinite(embeddings).all():
                raise ValueError("Embeddings contain NaN or Inf values")

            # 4) D2H: cpu() 时间
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t4 = time.perf_counter()

            embedding_batches.append(embeddings.detach().cpu())

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t5 = time.perf_counter()

            wait = (t1 - t0) 
            h2d = (t3 - t2) 
            d2h = (t5 - t4) 
            fwd = fwd_ms / 1000.0
            bsz = int(batch["attention_mask"].shape[0])
            seq = int(batch["attention_mask"].shape[1])

            pbar.write(
                f"[{message} step={step}] "
                f"wait={wait:.3f}s "
                f"h2d={h2d:.3f}s "
                f"fwd={fwd:.3f}s "
                f"d2h={d2h:.3f}s "
                f"bsz={bsz} seq={seq}"
            )

            step += 1
            pbar.update(1)

        pbar.close()
        return _pad_and_stack_embeddings(embedding_batches)

    def forward_queries(self, queries: List[str], batch_size: int = 8) -> torch.Tensor:
        """Forward text queries and extract embeddings.

        Args:
            queries: List of query strings.
            batch_size: Batch size for processing.

        Returns:
            Tensor of query embeddings with shape (num_queries, max_seq_len, hidden_dim).
        """
        if isinstance(queries, DataLoader):
            dataset = queries.dataset
        else:
            dataset = ListDataset[str](queries)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.process_queries,
            shuffle=False,
            drop_last=False,
        )
        return self._extract_embeddings(dataloader=dataloader, is_query=True)

    def forward_documents(
        self, corpus: List[Dict], batch_size: int = 8
    ) -> torch.Tensor:
        """Forward documents (image + text) and extract embeddings.

        Args:
            corpus: List of dicts with "image" and "text" keys.
            batch_size: Batch size for processing.

        Returns:
            Tensor of document embeddings with shape (num_docs, max_seq_len, hidden_dim).
        """
        images = []
        texts = []
        for doc in corpus:
            text = doc.get("text", "")
            image = doc.get("image")
            if image is not None and image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)
            texts.append(text)

        dataset = Dataset.from_dict({"image": images, "text": texts})
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.process_documents,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        return self._extract_embeddings(dataloader=dataloader, is_query=False)

    def forward_images(
        self, images: List, batch_size: int = 8, **kwargs
    ) -> torch.Tensor:
        """Forward images as image-only documents.

        Args:
            images: List of PIL Images.
            batch_size: Batch size for processing.

        Returns:
            Tensor of image embeddings.
        """
        corpus = [{"image": image, "text": ""} for image in images]
        return self.forward_documents(corpus, batch_size)

    def forward_passages(
        self, images: List, batch_size: int = 8, **kwargs
    ) -> torch.Tensor:
        """Forward passages as image-only documents (alias for forward_images)."""
        return self.forward_images(images, batch_size, **kwargs)


class ColBERTScoringMixin:
    """Mixin providing ColBERT MaxSim scoring methods."""

    def colbert_score(
        self,
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Compute ColBERT MaxSim scores between queries and passages.

        Args:
            qs: Query embeddings - tensor or list of tensors.
            ps: Passage embeddings - tensor or list of tensors.
            batch_size: Batch size for scoring computation.
            device: Device to run computation on.

        Returns:
            Score matrix of shape (num_queries, num_passages).
        """
        if batch_size is None:
            batch_size = 128
        if device is None:
            device = next(self.parameters()).device

        if isinstance(qs, torch.Tensor):
            qs = [qs[i] for i in range(qs.shape[0])]
        if isinstance(ps, torch.Tensor):
            ps = [ps[i] for i in range(ps.shape[0])]

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_slice = qs[i : i + batch_size]
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                [q.to(device) for q in qs_slice], batch_first=True, padding_value=0
            )
            for j in range(0, len(ps), batch_size):
                ps_slice = ps[j : j + batch_size]
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    [p.to(device) for p in ps_slice], batch_first=True, padding_value=0
                )
                scores_batch.append(
                    torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                    .max(dim=3)[0]
                    .sum(dim=2)
                )
            scores_batch = torch.cat(scores_batch, dim=1)
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        return scores

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        """Compute ColBERT MaxSim scores between queries and passages.

        Args:
            query_embeddings: Query embeddings.
            passage_embeddings: Passage embeddings.
            batch_size: Batch size for scoring computation.

        Returns:
            Score matrix of shape (num_queries, num_passages).
        """
        if isinstance(query_embeddings, list):
            if len(query_embeddings[0].shape) == 2:
                query_embeddings = [q.unsqueeze(0) for q in query_embeddings]
            query_embeddings = _pad_and_stack_embeddings(query_embeddings)
        if isinstance(passage_embeddings, list):
            if len(passage_embeddings[0].shape) == 2:
                passage_embeddings = [p.unsqueeze(0) for p in passage_embeddings]
            passage_embeddings = _pad_and_stack_embeddings(passage_embeddings)

        return self.colbert_score(
            query_embeddings, passage_embeddings, batch_size or 128
        )


class Qwen3VLNemotronEmbedModel(
    EmbeddingMixin, ColBERTScoringMixin, Qwen3VLNemotronEmbedForConditionalGeneration
):
    """Qwen3VLNemotronEmbed: Vision-Language Embedding Model.

    A bidirectional vision-language model for document retrieval and embedding tasks.
    Based on Qwen3VL architecture with bidirectional attention for embedding extraction.

    Features:
    - ColBERT MaxSim scoring (get_scores, colbert_score)
    - High-level embedding methods (forward_queries, forward_documents, forward_images)
    - Automatic processor loading for query/document processing

    Example:
        >>> model = AutoModel.from_pretrained("nvidia/qwen3vl-nemotron-embed-4b", trust_remote_code=True)
        >>> query_embeddings = model.forward_queries(["What is machine learning?"])
        >>> doc_embeddings = model.forward_documents([{"image": img, "text": "ML explanation"}])
        >>> scores = model.get_scores(query_embeddings, doc_embeddings)
    """

    config_class = Qwen3VLNemotronEmbedConfig
