import os
import json
import torch
import numpy as np
import faiss
from argparse import ArgumentParser
import datasets
from tqdm import tqdm

from gme_dataset import GMEQueryDataset, GMECorpusDataset
from gme_model import GmeQwen2VL

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
            
            retrieved_ids, retrieved_images, retrieved_contexts = [], [], []
            
            for rank, idx in enumerate(top_indices):
                if idx == -1: continue
                item = corpus_lookup[idx]
                retrieved_ids.append(item['p_id'])
                retrieved_images.append(item.get('page_path', ''))
                retrieved_contexts.append(item.get('context', ''))
                
            f.write(json.dumps({
                "q_id": q_sample['q_id'], "question": q_sample['question'],
                "answer": q_sample.get('answer', ''), "target_ids": q_sample.get('target_ids', []),
                "retrieved_ids": retrieved_ids, "retrieved_images": retrieved_images,
                "retrieved_context": retrieved_contexts,
                "scores": [float(s) for s in top_scores]
            }) + "\n")
            
            # Use full retrieved list for metrics calculation
            full_retrieved_ids = [corpus_lookup[idx]['p_id'] for idx in indices[i] if idx != -1]
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
    parser.add_argument("--root_data", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/Benchmarks/processed_data")
    parser.add_argument("--output_root", type=str, default="/share/project/liuze/projects/contextual_mmemb/results/GME")
    parser.add_argument("--model_path", type=str, default="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--tasks", nargs='+', default=None)
    args = parser.parse_args()
    
    # 1. Init Model
    model = GmeQwen2VL(model_path=args.model_path)
    
    # Get Benchmarks
    tasks = []
    for subdir, _, files in os.walk(args.root_data):
        if "query.jsonl" in files:
            corpus_path = os.path.join(subdir, "corpus_with_context/contextual_corpus_local_4_images.jsonl")
            query_path = os.path.join(subdir, "query.jsonl")
            if os.path.exists(corpus_path):
                task_name = os.path.relpath(subdir, args.root_data)
                tasks.append({"name": task_name, "query_path": query_path, "corpus_path": corpus_path})
    
    if args.tasks:
        tasks = [t for t in tasks if any(req_task in t['name'] for req_task in args.tasks)]
        print(f"Filtered tasks ({len(tasks)}): {[t['name'] for t in tasks]}")
    else:
        print(f"Running all {len(tasks)} tasks.")

    for task in tasks:
        try:
            print(f"\nProcessing {task['name']}...")
            raw_queries = datasets.load_dataset('json', data_files=task['query_path'], split='train')
            raw_corpus = datasets.load_dataset('json', data_files=task['corpus_path'], split='train')
            
            query_ds = GMEQueryDataset(raw_queries)
            corpus_ds = GMECorpusDataset(raw_corpus)
            
            # 2. Encode Queries
            print("Encoding Queries...")
            q_embs = model.get_text_embeddings(
                texts=[item['text'] for item in query_ds], 
                batch_size=args.batch_size,
                instruction="Find a relevant document to the given query."
            )
            
            # 3. Encode Corpus (Images)
            print("Encoding Corpus...")
            c_embs = model.get_image_embeddings(
                images=[item['image'] for item in corpus_ds],
                batch_size=args.batch_size
            )
            
            # 4. Retrieval (Dense)
            index = faiss.index_factory(c_embs.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
            index.add(c_embs.numpy().astype(np.float32))
            scores, indices = index.search(q_embs.numpy().astype(np.float32), args.k)
            
            # 5. Save
            final_out_dir = os.path.join(args.output_root, "default", task['name'])
            save_results(final_out_dir, raw_queries, indices, scores, raw_corpus)
            
        except Exception as e:
            print(f"Error processing {task['name']}: {e}")
            continue

if __name__ == "__main__":
    main()
