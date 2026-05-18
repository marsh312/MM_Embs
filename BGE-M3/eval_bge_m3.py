import os
import json
import torch
import numpy as np
import faiss
from argparse import ArgumentParser
import logging

# Patch transformers.utils.import_utils.is_torch_fx_available for FlagEmbedding compatibility
# This resolves ImportError: cannot import name 'is_torch_fx_available'
try:
    import transformers.utils.import_utils
    if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
        def is_torch_fx_available():
            return True
        transformers.utils.import_utils.is_torch_fx_available = is_torch_fx_available
except ImportError:
    pass

from bge_m3_dataset import BGEM3QueryDataset, BGEM3CorpusDataset
from bge_m3_model import BGEM3TextEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# Evaluation Metrics Logic (Adapted from eval_rzen.py)
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

def main():
    parser = ArgumentParser()
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus.jsonl")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query.jsonl")
    parser.add_argument("--pages_path", type=str, required=True, help="Path to pages.jsonl (for text lookup)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_path", type=str, default="/share/project/shared_models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=200)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Init Model
    logger.info("Initializing model...")
    model = BGEM3TextEmbedding(model_name=args.model_path)
    
    # 2. Load Datasets
    logger.info("Loading datasets...")
    query_ds = BGEM3QueryDataset(args.query_path)
    corpus_ds = BGEM3CorpusDataset(args.corpus_path, args.pages_path)
    
    # 3. Encode Queries
    logger.info("Encoding queries...")
    q_embs = model.encode_queries(query_ds, batch_size=args.batch_size)
    
    # 4. Encode Corpus
    logger.info("Encoding corpus...")
    c_embs = model.encode_corpus(corpus_ds, batch_size=args.batch_size)
    
    # 5. Retrieval using FAISS
    logger.info("Running retrieval...")
    index = faiss.index_factory(c_embs.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(c_embs.astype(np.float32))
    scores, indices = index.search(q_embs.astype(np.float32), args.k)
    
    # 6. Process Results and Compute Metrics
    logger.info("Processing results...")
    results_for_metrics = []
    
    # We need to map corpus indices back to p_ids
    corpus_p_ids = [item['p_id'] for item in corpus_ds]
    
    # Prepare retrieval output
    retrieval_output_path = os.path.join(args.output_dir, "retrieval_output.jsonl")
    with open(retrieval_output_path, 'w') as f:
        for i in range(len(query_ds)):
            q_sample = query_ds[i]
            top_indices = indices[i][:10] # Top 10 for detailed output
            top_scores = scores[i][:10]
            
            retrieved_ids = []
            retrieved_texts = []
            
            for rank, idx in enumerate(top_indices):
                if idx == -1: continue
                # Get corpus item to find p_id
                # Note: corpus_ds[idx] performs lookup which might be slow if repeated.
                # But here we only do it for top 10.
                corpus_item = corpus_ds[idx] 
                p_id = corpus_item['p_id']
                text = corpus_item['text']
                
                retrieved_ids.append(p_id)
                retrieved_texts.append(text[:200] + "...") # Truncate for log
                
            f.write(json.dumps({
                "q_id": q_sample['q_id'], 
                "question": q_sample['q_text'],
                "target_ids": q_sample['target_ids'],
                "retrieved_ids": retrieved_ids,
                "retrieved_texts": retrieved_texts,
                "scores": top_scores.tolist()
            }) + "\n")
            
            # For metrics, we use the full k retrieved items
            full_retrieved_ids = []
            for idx in indices[i]:
                if idx != -1:
                    full_retrieved_ids.append(corpus_p_ids[idx])
            
            results_for_metrics.append((full_retrieved_ids, q_sample['target_ids']))
            
    # 7. Compute Metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics([x[0] for x in results_for_metrics], [x[1] for x in results_for_metrics])
    
    metrics_path = os.path.join(args.output_dir, "retrieval_results.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {metrics_path}")
    print(f"NDCG@10: {metrics.get('NDCG@10', 0.0):.4f}")

if __name__ == "__main__":
    main()
