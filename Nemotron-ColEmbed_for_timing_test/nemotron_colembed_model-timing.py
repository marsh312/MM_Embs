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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


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
        clean_timing_log: bool = True,
        suppress_model_progress: bool = True,
        timing_log_path: Optional[str] = None,
    ):
        self.clean_timing_log = clean_timing_log
        self.suppress_model_progress = suppress_model_progress
        self._timing_log_fh = None
        self.timing_log_path = timing_log_path or os.environ.get("NEMOTRON_TIMING_LOG_PATH")
        if self.timing_log_path:
            os.environ["NEMOTRON_TIMING_LOG_PATH"] = self.timing_log_path
            timing_dir = os.path.dirname(self.timing_log_path)
            if timing_dir:
                os.makedirs(timing_dir, exist_ok=True)
            self._timing_log_fh = open(self.timing_log_path, "a", encoding="utf-8", buffering=1)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        ).eval()

    def __del__(self):
        if self._timing_log_fh:
            self._timing_log_fh.close()

    def _log_timing(self, event: str, **fields) -> None:
        if not self.clean_timing_log:
            return
        payload = {"ts": _utc_now_iso(), "event": event}
        payload.update(fields)
        line = json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        if self._timing_log_fh:
            self._timing_log_fh.write(line + "\n")

    def forward_queries(self, queries: List[str], batch_size: int = 32) -> torch.Tensor:
        total_start = time.perf_counter()
        self._log_timing(
            "query_encode_start",
            num_queries=len(queries),
            batch_size=batch_size,
        )
        with _suppress_output(self.suppress_model_progress):
            embeddings = self.model.forward_queries(queries, batch_size=batch_size)
        self._log_timing(
            "query_encode_end",
            num_queries=len(queries),
            batch_size=batch_size,
            elapsed_s=round(time.perf_counter() - total_start, 6),
        )
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
        num_batches = (total_images + batch_size - 1) // batch_size if total_images > 0 else 0
        total_start = time.perf_counter()
        self._log_timing(
            "image_encode_start",
            total_images=total_images,
            batch_size=batch_size,
            num_batches=num_batches,
        )
        for batch_idx, i in enumerate(range(0, total_images, batch_size)):
            chunk = images_or_paths[i : i + batch_size]
            batch_start = time.perf_counter()
            load_start = time.perf_counter()
            if num_workers and len(chunk) > 1:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    loaded = list(executor.map(self._load_image_safe, chunk))
            else:
                loaded = [self._load_image_safe(x) for x in chunk]
            load_elapsed = time.perf_counter() - load_start
            forward_start = time.perf_counter()
            with _suppress_output(self.suppress_model_progress):
                emb = self.model.forward_images(loaded, batch_size=batch_size)
            forward_elapsed = time.perf_counter() - forward_start
            total_elapsed = time.perf_counter() - batch_start
            self._log_timing(
                "image_batch",
                batch_idx=batch_idx,
                load_s=round(load_elapsed, 6),
                forward_s=round(forward_elapsed, 6),
                total_s=round(total_elapsed, 6),
                num_batches=num_batches,
                start_index=i,
                batch_size=len(chunk),
            )
            batches.append(emb)
        self._log_timing(
            "image_encode_end",
            total_images=total_images,
            batch_size=batch_size,
            num_batches=num_batches,
            elapsed_s=round(time.perf_counter() - total_start, 6),
        )
        return _pad_and_stack_embeddings(batches)

    def get_scores(self, query_embeddings: Union[torch.Tensor, List[torch.Tensor]], passage_embeddings: Union[torch.Tensor, List[torch.Tensor]], batch_size: Optional[int] = 128) -> torch.Tensor:
        return self.model.get_scores(query_embeddings, passage_embeddings, batch_size=batch_size)
