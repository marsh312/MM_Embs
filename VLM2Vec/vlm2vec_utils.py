from __future__ import annotations

import logging
import math
import os
import sys
from typing import cast, Dict, List, Optional, Union, Any

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

# Add current directory to sys.path to ensure src can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS
from src.utils import batch_to_device

# --- Helper Functions from flag_dataset_vlm2vec.py ---

def validate_image_paths(img_list, base_image_dir):
    """
    验证图片路径是否存在，并返回有效的图片路径列表
    """
    valid_paths = []
    for img_path in img_list:
        if base_image_dir:
            full_path = os.path.join(base_image_dir, img_path)
        else:
            full_path = img_path
            
        if os.path.exists(full_path):
            valid_paths.append(img_path)
        else:
            print(f"Warning: Image path not found: {full_path}")
    
    return valid_paths

def smart_truncate_multimodal_sequence(text, images, tokenizer, max_length, image_token="<|image_pad|>"):
    """
    智能截断多模态序列
    """
    if not images:
        return text, images
    
    temp_inputs = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    temp_input_ids = temp_inputs.input_ids[0]
    
    if len(temp_input_ids) <= max_length:
        return text, images
    
    image_token_count = text.count(image_token)
    
    if image_token_count == 0:
        truncated_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
        truncated_text = tokenizer.decode(truncated_inputs.input_ids[0], skip_special_tokens=True)
        return truncated_text, []
    
    truncate_pos = max_length - 1
    
    if len(temp_input_ids) > max_length:
        keep_ratio = truncate_pos / len(temp_input_ids)
        keep_images = max(1, int(len(images) * keep_ratio))
        
        if keep_images < len(images):
            # print(f"Truncating from {len(images)} images to {keep_images} images due to max_length limit")
            text_parts = text.split(image_token)
            
            if len(text_parts) > keep_images + 1:
                new_text_parts = text_parts[:keep_images + 1]
                result_text = ""
                for i in range(len(new_text_parts)):
                    result_text += new_text_parts[i]
                    if i < keep_images:
                        result_text += image_token
                return result_text, images[:keep_images]
            else:
                return text, images[:keep_images]
    
    return text, images

# --- Dataset Classes ---

class MMIT_Dataset(Dataset):
    """多模态图文交错数据集，支持VLM2Vec格式"""
    def __init__(self, captions, image_ids, processor, image_dir=None) -> None:
        self.processor = processor
        self.image_dir = image_dir
        
        if isinstance(image_ids[0], list):
            self.image_paths = image_ids
        else:
            self.image_paths = [[img_id] for img_id in image_ids]

        self.captions = captions
    
    def __getitem__(self, item):
        images = []
        for img_path in self.image_paths[item]:
            if self.image_dir:
                full_path = os.path.join(self.image_dir, str(img_path))
            else:
                full_path = str(img_path)
            
            try:
                pil_data = Image.open(full_path).convert('RGB')
                images.append(pil_data)
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        caption = self.captions[item]
        return caption, images

    def __len__(self):
        return len(self.captions)

class MMIT_Collator:
    """多模态图文交错数据的collator，支持VLM2Vec格式"""
    def __init__(self, tokenizer, mmit_max_len, processor):
        self.mmit_max_len = mmit_max_len
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        captions = [f[0] for f in features]
        image_lists = [f[1] for f in features]
        
        processed_captions = []
        processed_images = []
        
        for caption, images in zip(captions, image_lists):
            truncated_caption, truncated_images = smart_truncate_multimodal_sequence(
                caption, images, self.tokenizer, self.mmit_max_len
            )
            
            image_token_count = truncated_caption.count(VLM_IMAGE_TOKENS[QWEN2_VL])
            
            if len(truncated_images) == 0:
                new_caption = truncated_caption.replace(VLM_IMAGE_TOKENS[QWEN2_VL], '')
                processed_captions.append(new_caption)
                processed_images.append([])
            elif image_token_count == len(truncated_images):
                processed_captions.append(truncated_caption)
                processed_images.append(truncated_images)
            elif image_token_count > len(truncated_images):
                if len(truncated_images) > 0:
                    repeated_images = truncated_images + [truncated_images[-1]] * (image_token_count - len(truncated_images))
                else:
                    repeated_images = []
                processed_captions.append(truncated_caption)
                processed_images.append(repeated_images)
            else:
                processed_captions.append(truncated_caption)
                processed_images.append(truncated_images[:image_token_count])
        
        try:
            text_only_samples = []
            multimodal_samples = []
            text_only_indices = []
            multimodal_indices = []
            
            for i, (cap, imgs) in enumerate(zip(processed_captions, processed_images)):
                if len(imgs) == 0:
                    text_only_samples.append(cap)
                    text_only_indices.append(i)
                else:
                    multimodal_samples.append((cap, imgs))
                    multimodal_indices.append(i)
            
            batch_results = [None] * len(processed_captions)
            
            if text_only_samples:
                text_inputs = self.processor(text=text_only_samples, return_tensors="pt", padding=True, truncation=True, max_length=self.mmit_max_len)
                for i, idx in enumerate(text_only_indices):
                    batch_results[idx] = {
                        'input_ids': text_inputs.input_ids[i:i+1],
                        'attention_mask': text_inputs.attention_mask[i:i+1]
                    }
            
            if multimodal_samples:
                mm_captions = [sample[0] for sample in multimodal_samples]
                mm_images = [sample[1] for sample in multimodal_samples]
                
                from transformers.image_utils import ChannelDimension
                mm_inputs = self.processor(text=mm_captions, images=mm_images, return_tensors="pt", padding=True, truncation=True, max_length=self.mmit_max_len, input_data_format=ChannelDimension.LAST)
                for i, idx in enumerate(multimodal_indices):
                    result = {
                        'input_ids': mm_inputs.input_ids[i:i+1],
                        'attention_mask': mm_inputs.attention_mask[i:i+1]
                    }
                    if hasattr(mm_inputs, 'pixel_values') and mm_inputs.pixel_values is not None:
                        result['pixel_values'] = mm_inputs.pixel_values[i:i+1]
                    if hasattr(mm_inputs, 'image_grid_thw') and mm_inputs.image_grid_thw is not None:
                        result['image_grid_thw'] = mm_inputs.image_grid_thw[i:i+1]
                    batch_results[idx] = result
            
            if batch_results:
                max_len = max(result['input_ids'].shape[-1] for result in batch_results)
                batch_size = len(batch_results)
                
                combined_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
                combined_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
                
                has_pixel_values = any('pixel_values' in result for result in batch_results if result is not None)
                if has_pixel_values:
                    pixel_shape = None
                    for result in batch_results:
                        if result is not None and 'pixel_values' in result:
                            pixel_shape = list(result['pixel_values'].shape)
                            break
                    if pixel_shape:
                        pixel_shape[0] = batch_size
                        combined_pixel_values = torch.zeros(pixel_shape)
                        combined_image_grid_thw = []
                
                for i, result in enumerate(batch_results):
                    if result is not None:
                        seq_len = result['input_ids'].shape[-1]
                        combined_input_ids[i, :seq_len] = result['input_ids'][0]
                        combined_attention_mask[i, :seq_len] = result['attention_mask'][0]
                        
                        if has_pixel_values and 'pixel_values' in result:
                            combined_pixel_values[i] = result['pixel_values'][0]
                            if 'image_grid_thw' in result:
                                combined_image_grid_thw.append(result['image_grid_thw'][0])
                            else:
                                combined_image_grid_thw.append(None)
                        elif has_pixel_values:
                            combined_image_grid_thw.append(None)
                
                final_inputs = {
                    'input_ids': combined_input_ids,
                    'attention_mask': combined_attention_mask
                }
                
                if has_pixel_values:
                    final_inputs['pixel_values'] = combined_pixel_values
                    final_inputs['image_grid_thw'] = combined_image_grid_thw
                
                return final_inputs
            else:
                return {
                    'input_ids': torch.zeros(1, 1, dtype=torch.long),
                    'attention_mask': torch.zeros(1, 1, dtype=torch.long)
                }
                
        except Exception as e:
            print(f"Error in MMIT_Collator: {e}")
            inputs = self.processor(text=processed_captions, return_tensors="pt", padding=True, truncation=True, max_length=self.mmit_max_len)
            return inputs

class Image_Dataset(Dataset):
    """纯图像数据集"""
    def __init__(self, image_ids, processor, image_dir=None) -> None:
        self.processor = processor
        self.image_dir = image_dir
        
        if isinstance(image_ids[0], list):
            self.image_paths = image_ids
        else:
            self.image_paths = [[img_id] for img_id in image_ids]

    def __getitem__(self, item):
        images = []
        for img_path in self.image_paths[item]:
            if self.image_dir:
                full_path = os.path.join(self.image_dir, str(img_path))
            else:
                full_path = str(img_path)
            
            try:
                pil_data = Image.open(full_path).convert('RGB')
                images.append(pil_data)
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        return images

    def __len__(self):
        return len(self.image_paths)

class Image_Collator:
    def __init__(self, tokenizer, image_max_len, processor):
        self.image_max_len = image_max_len
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        image_lists = features
        prompts = [f"Represent the given image. {VLM_IMAGE_TOKENS[QWEN2_VL]}" for _ in image_lists]
        
        from transformers.image_utils import ChannelDimension
        inputs = self.processor(text=prompts, images=image_lists, return_tensors="pt", padding=True, truncation=True, max_length=self.image_max_len, input_data_format=ChannelDimension.LAST)
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0) if len(inputs['pixel_values'].shape) == 3 else inputs['pixel_values']
        if 'image_grid_thw' in inputs:
             inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0) if len(inputs['image_grid_thw'].shape) == 1 else inputs['image_grid_thw']

        return inputs

# --- VLM2VecEmbedder ---

class VLM2VecEmbedder:
    def __init__(
        self,
        model_name: str = "VLM2Vec/VLM2Vec-V2.0",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length=2560,
        image_dir: str = None,
        **kwargs,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.image_dir = image_dir
        self.num_gpus = torch.cuda.device_count()
        
        self.model_args = ModelArguments(
            model_name=model_name,
            pooling='last',
            normalize=True,
            model_backbone='qwen2_vl',
            lora=True
        )
        self.data_args = DataArguments()
        
        self.processor = load_processor(self.model_args, self.data_args)
        self.model = MMEBModel.load(self.model_args)
        self.model = self.model.to(device, dtype=torch.bfloat16)
        self.model.eval()
        
        self.tokenizer = self.processor.tokenizer
        self.normalize = True
        
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)
    
    def encode_queries(self, queries, batch_size: int=8, max_length: int=2560) -> np.ndarray:
        if hasattr(queries, '__getitem__') and hasattr(queries, '__len__'):
            return self._encode_mixed_queries(queries, batch_size=batch_size, max_length=max_length)
        else:
            return self._encode_dict_queries(queries, batch_size=batch_size, max_length=max_length)

    def _encode_mixed_queries(self, queries, batch_size: int=8, max_length: int=2560) -> np.ndarray:
        text_samples = []
        image_samples = []
        mmit_samples = []
        
        text_indices = []
        image_indices = []
        mmit_indices = []
        
        for i in range(len(queries)):
            sample = queries[i]
            q_text = sample["q_text"]
            q_image = sample["q_image"]
            
            has_text = q_text != ""
            has_image = isinstance(q_image, list) and len(q_image) > 0
            
            if has_text and not has_image:
                text_samples.append(q_text)
                text_indices.append(i)
            elif not has_text and has_image:
                image_samples.append(q_image)
                image_indices.append(i)
            elif has_text and has_image:
                mmit_samples.append((q_text, q_image))
                mmit_indices.append(i)
            else:
                # Fallback to text if empty? Or raise error
                text_samples.append(q_text)
                text_indices.append(i)
        
        print(f"Mixed query types detected: {len(text_samples)} text, {len(image_samples)} image, {len(mmit_samples)} multimodal")
        
        all_embeddings = [None] * len(queries)
        
        if text_samples:
            text_embeddings = self.encode_text(text_samples, batch_size=batch_size, max_length=max_length)
            for i, idx in enumerate(text_indices):
                all_embeddings[idx] = text_embeddings[i]
        
        if image_samples:
            image_embeddings = self.encode_image(image_samples, batch_size=batch_size, max_length=max_length)
            for i, idx in enumerate(image_indices):
                all_embeddings[idx] = image_embeddings[i]
        
        if mmit_samples:
            mmit_texts = [sample[0] for sample in mmit_samples]
            mmit_images = [sample[1] for sample in mmit_samples]
            mmit_embeddings = self.encode_mm_it(mmit_texts, mmit_images, batch_size=batch_size, max_length=max_length)
            for i, idx in enumerate(mmit_indices):
                all_embeddings[idx] = mmit_embeddings[i]
        
        return np.array(all_embeddings)

    def encode_corpus(self, corpus, batch_size: int=8, max_length: int=2560) -> np.ndarray:
        if hasattr(corpus, '__getitem__') and hasattr(corpus, '__len__'):
            return self._encode_mixed_corpus(corpus, batch_size=batch_size, max_length=max_length)
        else:
             # Should not be used in our case but for completeness
            return np.array([])

    def _encode_mixed_corpus(self, corpus, batch_size: int=8, max_length: int=2560) -> np.ndarray:
        text_samples = []
        image_samples = []
        mmit_samples = []
        
        text_indices = []
        image_indices = []
        mmit_indices = []
        
        for i in range(len(corpus)):
            item = corpus[i]
            text = item["text"]
            image = item["image"]
            
            has_text = text != ""
            has_image = isinstance(image, list) and len(image) > 0
            
            if has_text and not has_image:
                text_samples.append(text)
                text_indices.append(i)
            elif not has_text and has_image:
                image_samples.append(image)
                image_indices.append(i)
            elif has_text and has_image:
                mmit_samples.append((text, image))
                mmit_indices.append(i)
            else:
                 # Fallback
                text_samples.append(text)
                text_indices.append(i)
        
        print(f"Mixed corpus types detected: {len(text_samples)} text, {len(image_samples)} image, {len(mmit_samples)} multimodal")
        
        all_embeddings = [None] * len(corpus)
        
        if text_samples:
            text_embeddings = self.encode_text(text_samples, batch_size=batch_size, max_length=max_length)
            for i, idx in enumerate(text_indices):
                all_embeddings[idx] = text_embeddings[i]
        
        if image_samples:
            image_embeddings = self.encode_image(image_samples, batch_size=batch_size, max_length=max_length)
            for i, idx in enumerate(image_indices):
                all_embeddings[idx] = image_embeddings[i]
        
        if mmit_samples:
            mmit_texts = [sample[0] for sample in mmit_samples]
            mmit_images = [sample[1] for sample in mmit_samples]
            mmit_embeddings = self.encode_mm_it(mmit_texts, mmit_images, batch_size=batch_size, max_length=max_length)
            for i, idx in enumerate(mmit_indices):
                all_embeddings[idx] = mmit_embeddings[i]
        
        return np.array(all_embeddings)

    @torch.no_grad()
    def encode_text(self, sentences: Union[List[str], str], batch_size: int=8, max_length: int=2560) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Text Embeddings", disable=len(sentences)<128):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.processor(text=sentences_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = batch_to_device(inputs, self.device)
            
            embeddings = self.model(tgt=inputs)["tgt_reps"]
            embeddings = cast(torch.Tensor, embeddings)
            
            embeddings = embeddings.float()
            all_embeddings.append(embeddings.cpu().numpy())

        if len(all_embeddings) > 0:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            return np.array([])
            
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings
    
    @torch.no_grad()
    def encode_mm_it(self, captions: Union[List[str], str], image_ids: Union[List[List[str]], List[str], str],  batch_size: int=8, max_length: int=2560) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        
        input_was_string = False
        if isinstance(captions, str):
            captions = [captions]
            image_ids = [image_ids]
            input_was_string = True

        all_embeddings = []
        
        # Create dataset and dataloader for batch processing
        dataset = MMIT_Dataset(captions, image_ids, self.processor, self.image_dir)
        collator = MMIT_Collator(self.tokenizer, max_length, self.processor)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collator)
        
        for inputs in tqdm(dataloader, desc="Inference Mm-it Embeddings", disable=len(captions)<128):
            inputs = batch_to_device(inputs, self.device)
            
            embeddings = self.model(qry=inputs)["qry_reps"]
            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.float()
            all_embeddings.append(embeddings.cpu().numpy())

        if len(all_embeddings) > 0:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            return np.array([])
            
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    @torch.no_grad()
    def encode_image(self, image_ids: Union[List[List[str]], List[str], str],  batch_size: int=8, max_length: int=2560) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus

        all_embeddings = []
        image_dataset = Image_Dataset(image_ids=image_ids, 
                                     processor=self.processor,
                                     image_dir=self.image_dir
                                     )
        image_collator = Image_Collator(self.tokenizer, image_max_len=max_length, processor=self.processor)

        image_dataloader = DataLoader(dataset=image_dataset, 
                                      collate_fn=image_collator, 
                                      num_workers=8, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,)

        for batch_data in tqdm(image_dataloader, desc="Inference Image Embeddings"):
            inputs = batch_to_device(batch_data, self.device)
            embeddings = self.model(qry=inputs)["qry_reps"]
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        if len(all_embeddings) > 0:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            return np.array([])
            
        return all_embeddings
