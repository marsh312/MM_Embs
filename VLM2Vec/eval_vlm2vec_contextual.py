import os
import json
import torch
import numpy as np
import faiss
from argparse import ArgumentParser
import datasets
import sys

# Ensure we can import vlm2vec_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from vlm2vec_utils import VLM2VecEmbedder

# --------------------------
# Evaluation Metrics Logic (Copied from eval_rzen_contextual.py)
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
    # Convert corpus_data to list if it's not indexable by integer index efficiently
    # raw_corpus from datasets.load_dataset supports integer indexing
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

            f.write(json.dumps({
                "q_id": q_sample['q_id'], 
                "question": q_sample['question'],
                "answer": q_sample.get('answer', ''), 
                "target_ids": q_sample.get('target_ids', []),
                "retrieved_ids": retrieved_ids, 
                "retrieved_images": retrieved_images,
                "scores": retrieved_scores
            }) + "\n")
            
            # Use full retrieved list for metrics calculation
            full_retrieved_ids = [corpus_lookup[int(idx)]['p_id'] for idx in indices[i] if idx != -1]
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
    parser.add_argument("--model_name", type=str, default="VLM2Vec/VLM2Vec-V2.0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--k", type=int, default=200)
    args = parser.parse_args()
    
    # Paths
    base_data_dir = "/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas"
    query_path = os.path.join(base_data_dir, args.source, "query.jsonl")
    corpus_path = os.path.join(base_data_dir, args.source, "corpus.jsonl")
    # Changed output dir to VLM2Vec
    output_dir = f"/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/VLM2Vec/{args.source}"
    
    # 1. Init Model
    print(f"Initializing VLM2Vec model: {args.model_name}...")
    model = VLM2VecEmbedder(model_name=args.model_name)
    
    print(f"Processing {args.source}...")
    print(f"Query Path: {query_path}")
    print(f"Corpus Path: {corpus_path}")
    print(f"Output Dir: {output_dir}")

    try:
        # Load queries
        with open(query_path, 'r') as f:
            raw_queries = [json.loads(line) for line in f]
        
        # Load corpus
        # datasets.load_dataset is efficient for large files
        raw_corpus = datasets.load_dataset('json', data_files=corpus_path, split='train')

        # 2. Encode Queries (Text with instruction)
        query_texts = []
        for q in raw_queries:
            # Add instruction
            text = f"Find a relevant document to the given query. {q['question']}"
            query_texts.append(text)
            
        print(f"Encoding {len(query_texts)} queries...")
        q_embs = model.encode_text(
            sentences=query_texts,
            batch_size=args.batch_size
        )

        # 3. Encode Corpus (Images with instruction handled by model wrapper)
        # Prepare list of image paths
        corpus_image_paths = [item['page_path'] for item in raw_corpus]
        
        print(f"Encoding {len(corpus_image_paths)} corpus images...")
        c_embs = model.encode_image(
            image_ids=corpus_image_paths,
            batch_size=args.batch_size
        )
        
        # 4. Retrieval
        print("Building index and searching...")
        index = faiss.index_factory(c_embs.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(c_embs.astype(np.float32))
        scores, indices = index.search(q_embs.astype(np.float32), args.k)

        # 5. Save
        print("Saving results...")
        save_results(output_dir, raw_queries, indices, scores, raw_corpus)
        print("Done.")

    except Exception as e:
        print(f"Error processing {args.source}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
