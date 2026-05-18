import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from bm25_dataset import BM25CorpusDataset, BM25QueryDataset
from bm25_model import BM25Okapi

def compute_metrics(qrels, results, k_values=[1, 5, 10]):
    """
    Compute MRR, Recall, and NDCG at various k.
    qrels: dict {qid: {pid: 1, ...}}
    results: dict {qid: [pid1, pid2, ...]} (sorted list of retrieved pids)
    """
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
        
        # Sort retrieved_pids is already done by score
        
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
            
            # Compute DCG
            for rank, pid in enumerate(retrieved_k):
                if pid in target_pids_dict:
                    dcg += 1.0 / np.log2(rank + 2)
            
            # Compute IDCG
            # Ideal ranking has all relevant docs at the top
            num_relevant = len(target_pids)
            # For this task, usually we have fewer relevant docs than k, or more.
            # We take the top min(k, num_relevant) slots as 1, others 0
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
    parser = argparse.ArgumentParser(description="BM25 Evaluation")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus.jsonl")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query.jsonl")
    parser.add_argument("--pages_path", type=str, required=True, help="Path to pages.jsonl")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--top_k", type=int, default=100, help="Number of documents to retrieve")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Datasets
    print("Loading datasets...")
    corpus_dataset = BM25CorpusDataset(args.corpus_path, args.pages_path)
    query_dataset = BM25QueryDataset(args.query_path)
    
    # 2. Prepare Corpus for Indexing
    print("Preparing corpus texts...")
    corpus_texts = corpus_dataset.get_all_texts()
    corpus_pids = corpus_dataset.get_all_pids()
    
    # 3. Build BM25 Index
    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_texts)
    
    # 4. Retrieval
    print("Retrieving...")
    results = {} # qid -> [pid1, pid2, ...]
    
    output_path = os.path.join(args.output_dir, "retrieval_output.jsonl")
    f_out = open(output_path, 'w')
    
    for i in tqdm(range(len(query_dataset)), desc="Processing queries"):
        sample = query_dataset[i]
        qid = sample["q_id"]
        q_text = sample["q_text"]
        
        # Get top-k indices
        top_k_results = bm25.retrieve(q_text, top_k=args.top_k)
        
        # Map indices to pids
        retrieved_pids = []
        retrieved_texts = [] # Optional: save retrieved texts for inspection
        
        for doc_idx, score in top_k_results:
            pid = corpus_pids[doc_idx]
            retrieved_pids.append(pid)
            # retrieved_texts.append(corpus_texts[doc_idx]) 
        
        results[qid] = retrieved_pids
        
        # Write to file
        out_item = {
            "q_id": qid,
            "retrieved_ids": retrieved_pids
        }
        f_out.write(json.dumps(out_item) + "\n")
        
    f_out.close()
    
    # 5. Compute Metrics
    print("Computing metrics...")
    qrels = {}
    for i in range(len(query_dataset)):
        sample = query_dataset[i]
        qid = sample["q_id"]
        target_ids = sample["target_ids"]
        qrels[qid] = {str(tid): 1 for tid in target_ids}
        
    metrics = compute_metrics(qrels, results, k_values=[1, 5, 10, 20, 50])
    
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
