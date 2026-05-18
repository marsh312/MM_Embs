import os
import json
import torch
import numpy as np
import faiss
from argparse import ArgumentParser
import datasets
from tqdm import tqdm
from PIL import Image
import types

from bgevl_dataset import BGEVLQueryDataset, BGEVLCorpusDataset
from bgevl_model import AutoModel

# --------------------------
# Evaluation Metrics Logic
# --------------------------
def compute_metrics(preds, labels, cutoffs=[1, 5, 10, 20, 50, 100]):
    assert len(preds) == len(labels)
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff: mrrs[k] += 1 / i
                jump = True
            if jump: break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs): metrics[f"MRR@{cutoff}"] = mrrs[i]

    # Recall & Easy Recall
    recalls, easy_recalls = np.zeros(len(cutoffs)), np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        if not isinstance(label, list): label = [label]
        for k, cutoff in enumerate(cutoffs):
            cnt = len(np.intersect1d(label, pred[:cutoff]))
            recalls[k] += cnt / len(label)
            if cnt > 0: easy_recalls[k] += 1
    recalls /= len(preds); easy_recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        metrics[f"Recall@{cutoff}"] = recalls[i]
    for i, cutoff in enumerate(cutoffs):
        metrics[f"Easy_Recall@{cutoff}"] = easy_recalls[i]

    # NDCG
    ndcgs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        if not isinstance(label, list): label = [label]
        for k, cutoff in enumerate(cutoffs):
            dcg, idcg = 0.0, 0.0
            for i, item in enumerate(pred[:cutoff]):
                if item in label: dcg += 1.0 / np.log2(i + 2)
            for i in range(min(len(label), cutoff)): idcg += 1.0 / np.log2(i + 2)
            ndcgs[k] += (dcg / idcg) if idcg > 0 else 0.0
    ndcgs /= len(preds)
    for i, cutoff in enumerate(cutoffs): metrics[f"NDCG@{cutoff}"] = ndcgs[i]
    return metrics

def save_results(output_dir, raw_queries, indices, scores, corpus_data):
    os.makedirs(output_dir, exist_ok=True)
    corpus_lookup = corpus_data
    results_for_metrics = []
    
    with open(os.path.join(output_dir, "retrieval_output.jsonl"), 'w') as f:
        for i, q_sample in enumerate(raw_queries):
            top_indices = indices[i][:10]
            top_scores = scores[i][:10]
            
            retrieved_ids, retrieved_images = [], []
            retrieved_scores = []
            
            for rank, idx in enumerate(top_indices):
                if idx == -1: continue
                item = corpus_lookup[int(idx)]
                retrieved_ids.append(item['p_id'])
                retrieved_images.append(item.get('page_path', ''))
                retrieved_scores.append(float(top_scores[rank]))

            # Use full retrieved list for metrics calculation
            full_retrieved_ids = [corpus_lookup[int(idx)]['p_id'] for idx in indices[i] if idx != -1]
            
            target_ids = q_sample.get('target_ids', [])
            cur_ranks = {}
            for target_id in target_ids:
                if target_id in full_retrieved_ids:
                    rank = full_retrieved_ids.index(target_id)
                else:
                    rank = len(full_retrieved_ids)
                cur_ranks[target_id] = rank

            f.write(json.dumps({
                "q_id": q_sample['q_id'], 
                "question": q_sample['question'],
                "answer": q_sample.get('answer', ''), 
                "qa_type": q_sample.get('qa_type', ''),
                "source": q_sample.get('source', ""),
                "target_ids": target_ids,
                "api_analysis": q_sample.get('api_analysis', ''),
                "target_rank": q_sample.get('target_rank', -1),
                "context_rank": q_sample.get('context_rank', -1),
                "cur_ranks": cur_ranks,
                "retrieved_ids": retrieved_ids, 
                "retrieved_images": retrieved_images,
                "scores": retrieved_scores
            }) + "\n")
            
            results_for_metrics.append((full_retrieved_ids, q_sample.get('target_ids', [])))

    # Compute & Save Metrics
    metrics = compute_metrics([x[0] for x in results_for_metrics], [x[1] for x in results_for_metrics])
    with open(os.path.join(output_dir, "retrieval_results.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"  NDCG@10: {metrics.get('NDCG@10', 0.0):.4f}")

def encode_queries(model, queries, batch_size):
    all_embeddings = []
    # queries is list of dicts from BGEVLQueryDataset
    for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
        batch_queries = queries[i:i+batch_size]
        texts = [item['text'] for item in batch_queries]
        
        # Following user provided inference example for query processing
        task_instruction = "Retrieve the target image that answers the given question."
        
        # BGE-VL data_process expects 'text' as list for batch processing
        # q_or_c="q" for query
        with torch.no_grad():
            query_inputs = model.data_process(
                text=texts,
                images=None, # Text-only query as per requirement
                q_or_c="q",
                task_instruction=task_instruction
            )
            
            query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
            query_embs = torch.nn.functional.normalize(query_embs, dim=-1)
            all_embeddings.append(query_embs.cpu().numpy())
            
    if all_embeddings:
        return np.concatenate(all_embeddings, axis=0)
    return np.array([])

def encode_corpus(model, corpus_images, batch_size):
    all_embeddings = []
    for i in tqdm(range(0, len(corpus_images), batch_size), desc="Encoding Corpus"):
        batch_images = corpus_images[i:i+batch_size]
        # corpus_images is list of image paths
        
        with torch.no_grad():
            candidate_inputs = model.data_process(
                images=batch_images,
                q_or_c="c"
            )
            
            candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :]
            candi_embs = torch.nn.functional.normalize(candi_embs, dim=-1)
            all_embeddings.append(candi_embs.cpu().numpy())
            
    if all_embeddings:
        return np.concatenate(all_embeddings, axis=0)
    return np.array([])

def patched_data_process(self, images=None, text=None, q_or_c=None, task_instruction=None):
    if images is not None:
        _is_list = isinstance(images, list)
    elif text is not None:
        _is_list = isinstance(text, list)
    else:
        raise ValueError("images and text cannot be both None.")
    
    assert q_or_c in ["query", "candidate", "q", "c"]

    if not _is_list:
        text_input = self.prepare_text_input(images, text, q_or_c, task_instruction)
        text_input = [text_input]
        
        if images is not None:
            images = Image.open(images).resize((512,512)).convert("RGB")
            images = [images]
            inputs = self.processor(images=images, text=text_input, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=text_input, return_tensors="pt", padding=True)

    else:
        if text is None:
            text = [None] * len(images)
            
        # FIX: Handle images=None when text is a list
        images_iter = images if images is not None else [None] * len(text)
            
        text_input = [self.prepare_text_input(_image, _text, q_or_c, task_instruction) for _image, _text in zip(images_iter, text)]
        
        if images is not None:
            images = [Image.open(_image).resize((512,512)).convert("RGB") for _image in images]
            inputs = self.processor(images=images, text=text_input, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=text_input, return_tensors="pt", padding=True)
    
    inputs = inputs.to(self.device)

    return inputs

# --------------------------
# Main
# --------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Data source name (e.g., arxiv_cs)")
    parser.add_argument("--model_path", type=str, default="/share/project/shared_models/BGE-VL-v1.5-mmeb")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--base_data_dir", type=str, required=True)
    args = parser.parse_args()

    # Paths
    base_data_dir = args.base_data_dir
    query_path = os.path.join(base_data_dir, args.source, "query.jsonl")
    corpus_path = os.path.join(base_data_dir, args.source, "corpus.jsonl")
    
    output_dir = os.path.join(base_data_dir.replace("/testdatas", "/testdatas_results"), "BGE-VL", args.source)
    
    # 1. Init Model
    print(f"Loading model from {args.model_path}...")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16 )
    model.eval()
    model.cuda()
    model.set_processor(args.model_path)
    
    # Monkey-patch the data_process method to fix NoneType iteration error
    model.data_process = types.MethodType(patched_data_process, model)
    
    print(f"Processing {args.source}...")
    print(f"Query Path: {query_path}")
    print(f"Corpus Path: {corpus_path}")
    print(f"Output Dir: {output_dir}")

    try:
        # Load queries
        with open(query_path, 'r') as f:
            raw_queries = [json.loads(line) for line in f]
        
        raw_corpus = datasets.load_dataset('json', data_files=corpus_path, split='train')

        query_ds = BGEVLQueryDataset(raw_queries)
        corpus_ds = BGEVLCorpusDataset(raw_corpus)

        # 2. Encode Queries
        # query_ds is passed directly to handle batching inside encode_queries because we need 'text' field
        q_embs = encode_queries(
            model=model,
            queries=[query_ds[i] for i in range(len(query_ds))],
            batch_size=args.batch_size
        )

        # 3. Encode Corpus (Images)
        # corpus_ds image paths list
        c_embs = encode_corpus(
            model=model,
            corpus_images=[item['image'] for item in corpus_ds],
            batch_size=args.batch_size
        )

        # Retrieval
        index = faiss.index_factory(c_embs.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(c_embs.astype(np.float32))
        scores, indices = index.search(q_embs.astype(np.float32), args.k)

        # 4. Save
        save_results(output_dir, raw_queries, indices, scores, raw_corpus)
        print("Done.")

    except Exception as e:
        print(f"Error processing {args.source}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
