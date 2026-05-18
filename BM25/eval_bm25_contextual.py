import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from bm25_dataset import BM25CorpusDataset, BM25QueryDataset
from bm25_model import BM25Okapi

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
    """
    Saves retrieval results and computes metrics.
    indices: list of lists (retrieved indices per query) - for BM25, this might be list of pids if we change logic
    scores: list of lists (scores per query)
    """
    os.makedirs(output_dir, exist_ok=True)
    corpus_lookup = corpus_data # Should be list-accessible if we pass the dataset or list
    
    # If corpus_data is BM25CorpusDataset, it has corpus_items
    if hasattr(corpus_data, 'corpus_items'):
        corpus_lookup = corpus_data.corpus_items
        
    results_for_metrics = []
    
    with open(os.path.join(output_dir, "retrieval_output.jsonl"), 'w') as f:
        for i, q_sample in enumerate(raw_queries):
            # indices[i] contains indices into corpus_data for this query
            # For BM25, we might have retrieved directly pids or indices. 
            # We'll assume indices into corpus_lookup to be consistent.
            
            full_retrieved_ids = [corpus_lookup[idx]['p_id'] for idx in indices[i] if idx != -1]
            
            # Calculate Ranks (target_ids)
            cur_ranks = {}
            for tid in q_sample.get('target_ids', []):
                if tid in full_retrieved_ids:
                    cur_ranks[tid] = full_retrieved_ids.index(tid)
                else:
                    cur_ranks[tid] = len(full_retrieved_ids)
            
            top_indices = indices[i][:10]
            top_scores = scores[i][:10]
            
            retrieved_ids, retrieved_images = [], []
            retrieved_scores = []
            
            for rank, idx in enumerate(top_indices):
                if idx == -1: continue
                item = corpus_lookup[idx]
                retrieved_ids.append(item['p_id'])
                retrieved_images.append(item.get('page_path', ''))
                retrieved_scores.append(float(top_scores[rank]))

            f.write(json.dumps({
                "q_id": q_sample['q_id'], 
                "question": q_sample.get('question', q_sample.get('q_text', '')),
                "answer": q_sample.get('answer', ''), 
                "target_ids": q_sample.get('target_ids', []),
                "retrieved_ids": retrieved_ids, 
                "retrieved_images": retrieved_images,
                "scores": retrieved_scores,
                "api_analysis": q_sample.get('api_analysis', {}),
                "target_rank": q_sample.get('target_rank', {}),
                "context_rank": q_sample.get('context_rank', {}),
                "qa_type": q_sample.get('qa_type', ''),
                "source": q_sample.get('source', ""),
                "cur_ranks": cur_ranks
            }) + "\n")
            
            results_for_metrics.append((full_retrieved_ids, q_sample.get('target_ids', [])))

    # Compute & Save Metrics
    metrics = compute_metrics([x[0] for x in results_for_metrics], [x[1] for x in results_for_metrics])
    with open(os.path.join(output_dir, "retrieval_results.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"  NDCG@10: {metrics.get('NDCG@10', 0.0):.4f}")

def main():
    parser = argparse.ArgumentParser(description="BM25 Evaluation")
    parser.add_argument("--source", type=str, required=True, help="Data source name (e.g., arxiv_cs)")
    parser.add_argument("--k", type=int, default=200, help="Number of documents to retrieve")
    parser.add_argument("--base_data_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Paths
    base_data_dir = args.base_data_dir
    pdf_data_dir = "/share/project/liuze/projects/contextual_mmemb/data/pdf_datas"
    query_path = os.path.join(base_data_dir, args.source, "query.jsonl")
    corpus_path = os.path.join(base_data_dir, args.source, "corpus.jsonl")
    pages_path = os.path.join(pdf_data_dir, args.source, "rawdata", "pages.jsonl")
    
    output_dir = os.path.join(base_data_dir.replace("/testdatas", "/testdatas_results"), "BM25", args.source)
    
    print(f"Processing {args.source}...")
    
    print("Loading datasets...")
    # Load queries
    with open(query_path, 'r') as f:
        raw_queries = [json.loads(line) for line in f]
    
    # Load corpus
    corpus_dataset = BM25CorpusDataset(corpus_path, pages_path)
    # Using existing method from BM25CorpusDataset
    corpus_texts = corpus_dataset.get_all_texts()
    
    # 2. Build Index
    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_texts)
    
    # 3. Retrieval
    print("Retrieving...")
    all_indices = []
    all_scores = []
    
    for q in tqdm(raw_queries, desc="Queries"):
        q_text = q.get('question', q.get('q_text', ''))
        top_k_results = bm25.retrieve(q_text, top_k=args.k)
        
        # top_k_results is list of (doc_idx, score)
        indices = [x[0] for x in top_k_results]
        scores = [x[1] for x in top_k_results]
        
        # Pad if less than k (though BM25 usually returns k if enough docs)
        if len(indices) < args.k:
            indices += [-1] * (args.k - len(indices))
            scores += [-1000.0] * (args.k - len(scores))
            
        all_indices.append(indices)
        all_scores.append(scores)
        
    # 4. Save
    print("Saving...")
    save_results(output_dir, raw_queries, all_indices, all_scores, corpus_dataset)
    print("Done.")

if __name__ == "__main__":
    main()
