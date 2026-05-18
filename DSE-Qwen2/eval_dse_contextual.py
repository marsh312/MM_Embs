import os
import json
import torch
import numpy as np
import faiss
from argparse import ArgumentParser
import datasets
from tqdm import tqdm

from dse_dataset import DSEQueryDataset, DSECorpusDataset
from dse_model import DSEQwenEmbed

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
            # Use full retrieved list for metrics calculation
            full_retrieved_ids = [corpus_lookup[idx]['p_id'] for idx in indices[i] if idx != -1]
            
            # Calculate Ranks (target_ids)
            cur_ranks = {}
            for tid in q_sample.get('target_ids', []):
                if tid in full_retrieved_ids:
                    cur_ranks[tid] = full_retrieved_ids.index(tid)
                else:
                    cur_ranks[tid] = len(full_retrieved_ids)

            top_indices = indices[i]
            top_scores = scores[i]
            
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
                "question": q_sample['question'],
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

# --------------------------
# Main
# --------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Data source name (e.g., arxiv_cs)")
    # parser.add_argument("--model_path", type=str, default="/share/project/shared_models/dse-qwen2-2b-mrl-v1")
    parser.add_argument("--model_path", type=str, default="/share/project/shared_models/dse-qwen2-2b-mrl-v1-safetensor")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--base_data_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Paths
    base_data_dir = args.base_data_dir
    query_path = os.path.join(base_data_dir, args.source, "query.jsonl")
    corpus_path = os.path.join(base_data_dir, args.source, "corpus.jsonl")
    
    output_dir = os.path.join(base_data_dir.replace("/testdatas", "/testdatas_results"), "DSE-Qwen2", args.source)
    
    # 1. Init Model
    print("Initializing Model...")
    model = DSEQwenEmbed(model_path=args.model_path, dimension=args.dimension)
    
    print(f"Processing {args.source}...")
    print(f"Query Path: {query_path}")
    print(f"Corpus Path: {corpus_path}")
    print(f"Output Dir: {output_dir}")

    try:
        # Load queries
        with open(query_path, 'r') as f:
            raw_queries = [json.loads(line) for line in f]
        
        raw_corpus = datasets.load_dataset('json', data_files=corpus_path, split='train')

        query_ds = DSEQueryDataset(raw_queries)
        corpus_ds = DSECorpusDataset(raw_corpus)

        # 2. Encode Queries (Text)
        print("Encoding Queries...")
        # Returns list of tensors [1, D] (Wait, DSEQwenEmbed.get_query_embeddings returns list of tensors)
        # Assuming it returns list of 1D tensors [D] or [1, D]
        # Let's verify dse_model.py
        # It returns `vecs[i]` where vecs is [B, D]. So it's [D].
        
        q_embs_list = model.get_query_embeddings(
            texts=[item['text'] for item in query_ds], 
            batch_size=args.batch_size
        )
        # Stack to tensor [N_q, D]
        q_embs = torch.stack(q_embs_list)

        # 3. Encode Corpus (Image)
        print("Encoding Corpus...")
        c_embs_list = model.get_image_embeddings(
            images=[item['image'] for item in corpus_ds],
            batch_size=args.batch_size
        )
        # Stack to tensor [N_c, D]
        c_embs = torch.stack(c_embs_list)

        # 4. Retrieval (Dense)
        print("Calculating Scores...")
        
        q_embs_np = q_embs.detach().cpu().float().numpy()
        c_embs_np = c_embs.detach().cpu().float().numpy()

        index = faiss.index_factory(c_embs_np.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(c_embs_np)
        scores, indices = index.search(q_embs_np, args.k)

        # 5. Save
        save_results(output_dir, raw_queries, indices, scores, raw_corpus)
        print("Done.")

    except Exception as e:
        print(f"Error processing {args.source}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
