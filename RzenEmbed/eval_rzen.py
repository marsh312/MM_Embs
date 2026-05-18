import os
import json
import torch
import numpy as np
import faiss
from argparse import ArgumentParser
import datasets # 保留datasets库用于读取jsonl

from rzen_dataset import RzenQueryDataset, RzenCorpusDataset
from rzen_model import RzenEmbed

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

# --------------------------
# Main Helper
# --------------------------
def get_benchmarks(root_folder):
    tasks = []
    for subdir, _, files in os.walk(root_folder):
        if "query.jsonl" in files:
            corpus_path = os.path.join(subdir, "corpus_with_context/contextual_corpus_local_4_images.jsonl")
            query_path = os.path.join(subdir, "query.jsonl")
            if os.path.exists(corpus_path):
                task_name = os.path.relpath(subdir, root_folder)
                tasks.append({"name": task_name, "query_path": query_path, "corpus_path": corpus_path})
    return tasks

def save_results(output_dir, raw_queries, indices, scores, corpus_data, score_components=None):
    os.makedirs(output_dir, exist_ok=True)
    corpus_lookup = corpus_data
    results_for_metrics = []
    
    with open(os.path.join(output_dir, "retrieval_output.jsonl"), 'w') as f:
        for i, q_sample in enumerate(raw_queries):
            top_indices = indices[i][:10]
            top_scores = scores[i][:10]
            
            retrieved_ids, retrieved_images, retrieved_contexts = [], [], []
            img_s_list, ctx_s_list, fus_s_list = [], [], []
            
            for rank, idx in enumerate(top_indices):
                if idx == -1: continue
                item = corpus_lookup[idx]
                retrieved_ids.append(item['p_id'])
                retrieved_images.append(item.get('page_path', ''))
                retrieved_contexts.append(item.get('context', ''))
                
                curr_score = float(top_scores[rank])
                if score_components and score_components['type'] == 'fusion':
                    # Calculate individual scores for the fusion case
                    s_img = float(np.dot(score_components['q_emb'][i], score_components['img_emb'][idx]))
                    s_ctx = float(np.dot(score_components['q_emb'][i], score_components['ctx_emb'][idx]))
                    img_s_list.append(s_img)
                    ctx_s_list.append(s_ctx)
                    fus_s_list.append(curr_score)
                elif score_components and score_components['type'] == 'image':
                    img_s_list.append(curr_score); ctx_s_list.append(0.0); fus_s_list.append(curr_score)
                elif score_components and score_components['type'] == 'context':
                    img_s_list.append(0.0); ctx_s_list.append(curr_score); fus_s_list.append(curr_score)
                else: # joint
                    img_s_list.append(0.0); ctx_s_list.append(0.0); fus_s_list.append(curr_score)

            f.write(json.dumps({
                "q_id": q_sample['q_id'], "question": q_sample['question'],
                "answer": q_sample.get('answer', ''), "target_ids": q_sample.get('target_ids', []),
                "retrieved_ids": retrieved_ids, "retrieved_images": retrieved_images,
                "retrieved_context": retrieved_contexts,
                "image_scores": img_s_list, "context_scores": ctx_s_list, "fusion_scores": fus_s_list
            }) + "\n")
            
            # Use full retrieved list for metrics calculation (indices[i] usually contains top-k, e.g., 100)
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
    parser.add_argument("--output_root", type=str, default="/share/project/liuze/projects/contextual_mmemb/results/RzenEmbed")
    parser.add_argument("--model_path", type=str, default="/share/project/shared_models/RzenEmbed")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--score_method", type=str, required=True, choices=['image', 'context', 'fusion', 'joint'])
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--tasks", nargs='+', default=None, help="指定要运行的 benchmark 名称关键字，例如: FinRAG SlideVQA")
    args = parser.parse_args()
    
    # 1. Init Model
    model = RzenEmbed(model_path=args.model_path)
    all_tasks = get_benchmarks(args.root_data)
    if args.tasks:
        # 只要任务名称中包含 args.tasks 中的任意一个关键词，就保留
        tasks = [t for t in all_tasks if any(req_task in t['name'] for req_task in args.tasks)]
        print(f"Filtered tasks ({len(tasks)}/{len(all_tasks)}): {[t['name'] for t in tasks]}")
    else:
        tasks = all_tasks
        print(f"Running all {len(tasks)} tasks: {[t['name'] for t in tasks]}")

    print(f"Running {args.score_method} on {len(tasks)} tasks.")

    for task in tasks:
        try:
            print(f"\nProcessing {task['name']}...")
            # Use datasets lib or json loading
            raw_queries = datasets.load_dataset('json', data_files=task['query_path'], split='train')
            raw_corpus = datasets.load_dataset('json', data_files=task['corpus_path'], split='train')

            # print(raw_queries[0])
            # print(raw_corpus[0])

            query_ds = RzenQueryDataset(raw_queries)
            corpus_ds = RzenCorpusDataset(raw_corpus)

            # 2. Encode Queries (Generic Instruction)
            q_embs = model.get_fused_embeddings(
                texts=[item['text'] for item in query_ds], 
                batch_size=args.batch_size,
                instruction="Find a relevant document to the given query." 
            )

            indices, scores, components = None, None, {}
            save_name = ""

            # 3. Encode Corpus & Retrieval
            if args.score_method == 'image':
                save_name = "image"
                c_embs = model.get_fused_embeddings(
                    images=[item['image'] for item in corpus_ds],
                    batch_size=args.batch_size,
                    instruction="Represent the given image."
                )
                index = faiss.index_factory(c_embs.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
                index.add(c_embs.astype(np.float32))
                scores, indices = index.search(q_embs.astype(np.float32), args.k)
                components = {'type': 'image'}

            elif args.score_method == 'context':
                save_name = "context"
                c_embs = model.get_fused_embeddings(
                    texts=[item['text'] for item in corpus_ds],
                    batch_size=args.batch_size,
                    instruction="Represent the given document context."
                )
                index = faiss.index_factory(c_embs.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
                index.add(c_embs.astype(np.float32))
                scores, indices = index.search(q_embs.astype(np.float32), args.k)
                components = {'type': 'context'}

            elif args.score_method == 'fusion':
                save_name = "image_score+context_score"
                print("  Encoding Image part...")
                c_embs_img = model.get_fused_embeddings(
                    images=[item['image'] for item in corpus_ds],
                    batch_size=args.batch_size,
                    instruction="Represent the given image."
                )
                print("  Encoding Context part...")
                c_embs_ctx = model.get_fused_embeddings(
                    texts=[item['text'] for item in corpus_ds],
                    batch_size=args.batch_size,
                    instruction="Represent the given document context."
                )
                
                # Fusion
                c_embs_fused = c_embs_img + (args.alpha * c_embs_ctx)
                
                index = faiss.index_factory(c_embs_fused.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
                index.add(c_embs_fused.astype(np.float32))
                scores, indices = index.search(q_embs.astype(np.float32), args.k)
                
                components = {
                    'type': 'fusion', 'q_emb': q_embs, 
                    'img_emb': c_embs_img, 'ctx_emb': c_embs_ctx, 'alpha': args.alpha
                }

            elif args.score_method == 'joint':
                save_name = "image+context_score"
                # Pass BOTH texts and images to get_fused_embeddings
                c_embs = model.get_fused_embeddings(
                    texts=[item['text'] for item in corpus_ds],
                    images=[item['image'] for item in corpus_ds],
                    batch_size=args.batch_size,
                    instruction="Represent the given image and document context." 
                )
                index = faiss.index_factory(c_embs.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
                index.add(c_embs.astype(np.float32))
                scores, indices = index.search(q_embs.astype(np.float32), args.k)
                components = {'type': 'joint'}

            # 4. Save
            final_out_dir = os.path.join(args.output_root, save_name, task['name'])
            save_results(final_out_dir, raw_queries, indices, scores, raw_corpus, components)
        except Exception as e:
            print(f"Error processing {task['name']}: {e}")
            continue

if __name__ == "__main__":
    main()