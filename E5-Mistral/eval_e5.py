import os
import json
import argparse
import numpy as np
import faiss
from tqdm import tqdm
import torch
from e5_dataset import E5QueryDataset, E5CorpusDataset
from e5_model import E5MistralModel

def compute_metrics(qrels, results, k_values=[1, 5, 10]):
    metrics = {}
    for k in k_values:
        metrics[f"MRR@{k}"] = 0.0
        metrics[f"Recall@{k}"] = 0.0
        metrics[f"NDCG@{k}"] = 0.0
    
    cnt = 0
    for qid, target_pids_dict in qrels.items():
        if qid not in results:
            continue
        cnt += 1
        retrieved_pids = results[qid]
        target_pids = list(target_pids_dict.keys())
        
        for k in k_values:
            retrieved_k = retrieved_pids[:k]
            
            # MRR
            mrr = 0.0
            for rank, pid in enumerate(retrieved_k):
                if pid in target_pids_dict:
                    mrr = 1.0 / (rank + 1)
                    break
            metrics[f"MRR@{k}"] += mrr
            
            # Recall
            recall = 0.0
            hits = sum(1 for pid in retrieved_k if pid in target_pids_dict)
            if len(target_pids) > 0:
                recall = hits / len(target_pids)
            metrics[f"Recall@{k}"] += recall
            
            # NDCG
            dcg = 0.0
            idcg = 0.0
            
            for rank, pid in enumerate(retrieved_k):
                if pid in target_pids_dict:
                    dcg += 1.0 / np.log2(rank + 2)
            
            num_relevant = len(target_pids)
            ideal_hits = min(k, num_relevant)
            for rank in range(ideal_hits):
                idcg += 1.0 / np.log2(rank + 2)
            
            ndcg = 0.0
            if idcg > 0:
                ndcg = dcg / idcg
            metrics[f"NDCG@{k}"] += ndcg

    if cnt > 0:
        for k, v in metrics.items():
            metrics[k] = v / cnt
            
    return metrics

def main():
    parser = argparse.ArgumentParser(description="E5-Mistral Evaluation")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus.jsonl")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query.jsonl")
    parser.add_argument("--pages_path", type=str, required=True, help="Path to pages.jsonl")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--model_path", type=str, default="/share/project/shared_models/models--intfloat--e5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1", help="Path to E5-Mistral model")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation (e.g. flash_attention_2, sdpa, eager)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference") # Reduced default batch size for 7B model
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading datasets...")
    corpus_dataset = E5CorpusDataset(args.corpus_path, args.pages_path)
    query_dataset = E5QueryDataset(args.query_path)
    
    # 2. Load Model
    print("Loading model...")
    model = E5MistralModel(model_name=args.model_path, attn_implementation=args.attn_implementation)
    
    # 3. Encode Corpus
    print("Encoding corpus...")
    corpus_texts = []
    corpus_pids = []
    for i in range(len(corpus_dataset)):
        item = corpus_dataset[i]
        corpus_texts.append(item["text"])
        corpus_pids.append(item["p_id"])
        
    corpus_embeddings = model.encode_corpus(corpus_texts, batch_size=args.batch_size)
    
    # 4. Build Index
    print("Building FAISS index...")
    d = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(corpus_embeddings)
    
    # 5. Encode Queries
    print("Encoding queries...")
    query_texts = []
    query_ids = []
    target_ids_map = {} # qid -> {pid: 1}
    
    for i in range(len(query_dataset)):
        item = query_dataset[i]
        query_texts.append(item["q_text"])
        query_ids.append(item["q_id"])
        target_ids_map[item["q_id"]] = {str(tid): 1 for tid in item["target_ids"]}
        
    query_embeddings = model.encode_queries(query_texts, batch_size=args.batch_size)
    
    # 6. Search
    print("Searching...")
    k = 100
    D, I = index.search(query_embeddings, k)
    
    # 7. Save Results & Compute Metrics
    print("Saving results and computing metrics...")
    results = {}
    output_path = os.path.join(args.output_dir, "retrieval_output.jsonl")
    
    with open(output_path, 'w') as f:
        for i in range(len(query_ids)):
            qid = query_ids[i]
            retrieved_indices = I[i]
            retrieved_pids = [corpus_pids[idx] for idx in retrieved_indices]
            results[qid] = retrieved_pids
            
            out_item = {
                "q_id": qid,
                "retrieved_ids": retrieved_pids
            }
            f.write(json.dumps(out_item) + "\n")
            
    metrics = compute_metrics(target_ids_map, results, k_values=[1, 5, 10, 20, 50])
    
    print("Metrics:")
    for k_metric, v in metrics.items():
        print(f"{k_metric}: {v:.4f}")
        
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
