import os
import json
import torch
import sys
import numpy as np
from argparse import ArgumentParser
import datasets
from nemotron_colembed_model import NemotronColEmbed

def compute_metrics(preds, labels, cutoffs=[1, 5, 10, 20, 50, 100]):
    assert len(preds) == len(labels)
    metrics = {}
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        metrics[f"MRR@{cutoff}"] = mrrs[i]
    recalls, easy_recalls = np.zeros(len(cutoffs)), np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        if not isinstance(label, list):
            label = [label]
        for k, cutoff in enumerate(cutoffs):
            cnt = len(np.intersect1d(label, pred[:cutoff]))
            recalls[k] += cnt / len(label)
            if cnt > 0:
                easy_recalls[k] += 1
    recalls /= len(preds)
    easy_recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        metrics[f"Recall@{cutoff}"] = recalls[i]
    for i, cutoff in enumerate(cutoffs):
        metrics[f"Easy_Recall@{cutoff}"] = easy_recalls[i]
    ndcgs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        if not isinstance(label, list):
            label = [label]
        for k, cutoff in enumerate(cutoffs):
            dcg, idcg = 0.0, 0.0
            for i, item in enumerate(pred[:cutoff]):
                if item in label:
                    dcg += 1.0 / np.log2(i + 2)
            for i in range(min(len(label), cutoff)):
                idcg += 1.0 / np.log2(i + 2)
            ndcgs[k] += (dcg / idcg) if idcg > 0 else 0.0
    ndcgs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        metrics[f"NDCG@{cutoff}"] = ndcgs[i]
    return metrics

def save_results(output_dir, raw_queries, indices, scores, corpus_data):
    os.makedirs(output_dir, exist_ok=True)
    corpus_lookup = corpus_data
    results_for_metrics = []
    with open(os.path.join(output_dir, "retrieval_output.jsonl"), "w") as f:
        for i, q_sample in enumerate(raw_queries):
            top_indices = indices[i]
            top_scores = scores[i]
            
            # Use full retrieved list for metrics calculation
            full_retrieved_ids = [corpus_lookup[idx]["p_id"] for idx in top_indices if idx != -1]
            
            # Calculate Ranks (target_ids)
            cur_ranks = {}
            for tid in q_sample.get('target_ids', []):
                if tid in full_retrieved_ids:
                    cur_ranks[tid] = full_retrieved_ids.index(tid)
                else:
                    cur_ranks[tid] = len(full_retrieved_ids)

            retrieved_ids, retrieved_images = [], []
            retrieved_scores = []
            for rank, idx in enumerate(top_indices):
                if idx == -1:
                    continue
                item = corpus_lookup[idx]
                retrieved_ids.append(item["p_id"])
                retrieved_images.append(item.get("page_path", ""))
                retrieved_scores.append(float(top_scores[rank]))
            f.write(
                json.dumps(
                    {
                        "q_id": q_sample["q_id"],
                        "question": q_sample["question"],
                        "answer": q_sample.get("answer", ""),
                        "target_ids": q_sample.get("target_ids", []),
                        "retrieved_ids": retrieved_ids,
                        "retrieved_images": retrieved_images,
                        "scores": retrieved_scores,
                        "api_analysis": q_sample.get('api_analysis', {}),
                        "target_rank": q_sample.get('target_rank', {}),
                        "context_rank": q_sample.get('context_rank', {}),
                        "qa_type": q_sample.get('qa_type', ''),
                         "source": q_sample.get('source', ""),
                        "cur_ranks": cur_ranks
                    }
                )
                + "\n"
            )
            results_for_metrics.append((full_retrieved_ids, q_sample.get("target_ids", [])))
    metrics = compute_metrics([x[0] for x in results_for_metrics], [x[1] for x in results_for_metrics])
    with open(os.path.join(output_dir, "retrieval_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  NDCG@10: {metrics.get('NDCG@10', 0.0):.4f}")

def parse_model_label(s):
    name = os.path.basename(s.rstrip("/"))
    parts = [p for p in name.split("-") if p]
    if "vl" in parts:
        i = parts.index("vl")
        seg = parts[i : i + 3]
        if len(seg) == 3:
            return "-".join([x.upper() for x in seg])
    return name




def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/share/project/shared_models/nemotron-colembed-vl-4b-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--score_batch_size", type=int, default=100)
    parser.add_argument("--corpus_chunk_size", type=int, default=1000)
    parser.add_argument("--score_inner_batch_size", type=int, default=128)
    parser.add_argument("--show_model_progress", action="store_true")
    parser.add_argument("--base_data_dir", type=str, required=True)
    args = parser.parse_args()
    base_data_dir = args.base_data_dir
    query_path = os.path.join(base_data_dir, args.source, "query.jsonl")
    corpus_path = os.path.join(base_data_dir, args.source, "corpus.jsonl")
    label = parse_model_label(args.model_path)
    output_dir = os.path.join(base_data_dir.replace("/testdatas", "/testdatas_results"), f"Nemotron-ColEmbed_{label}", args.source)
    print(f"Processing {args.source}...")
    print(f"Query Path: {query_path}")
    print(f"Corpus Path: {corpus_path}")
    print(f"Model: {args.model_path}")
    print(f"Output Dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model = NemotronColEmbed(
        model_path=args.model_path,
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    with open(query_path, "r") as f:
        raw_queries = [json.loads(line) for line in f]
    raw_corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    queries = [x["question"] for x in raw_queries]
    image_paths = [x["page_path"] for x in raw_corpus]
    print("Encoding Queries...", flush=True)
    query_embeddings = model.forward_queries(queries, batch_size=args.batch_size)
    print("Encoding Images...", flush=True)
    image_embeddings = model.forward_images(image_paths, batch_size=args.batch_size)
    print(query_embeddings.dtype, query_embeddings.device)
    print(image_embeddings.dtype, image_embeddings.device)
    print("Calculating Scores...", flush=True)
    all_indices = []
    all_scores = []
    score_batch_size = args.score_batch_size
    c_chunk_size = args.corpus_chunk_size
    device = next(model.model.parameters()).device

    for i in range(0, len(query_embeddings), score_batch_size):
        q_batch = query_embeddings[i : i + score_batch_size]
        q_batch = q_batch.to(device, non_blocking=True)
        running_top_vals = None
        running_top_inds = None
        for j in range(0, len(image_embeddings), c_chunk_size):
            c_chunk = image_embeddings[j : j + c_chunk_size]
            with torch.inference_mode():
                scores_chunk = model.get_scores(q_batch, c_chunk, batch_size=args.score_inner_batch_size)
            top_k_chunk = min(args.k, scores_chunk.shape[1])
            chunk_vals, chunk_local_inds = torch.topk(scores_chunk, top_k_chunk, dim=1)
            chunk_global_inds = chunk_local_inds + j
            if running_top_vals is None:
                running_top_vals = chunk_vals
                running_top_inds = chunk_global_inds
                continue
        
            merged_vals = torch.cat([running_top_vals, chunk_vals], dim=1)
            merged_inds = torch.cat([running_top_inds, chunk_global_inds], dim=1)
            keep_k = min(args.k, merged_vals.shape[1])
            running_top_vals, selected_pos = torch.topk(merged_vals, keep_k, dim=1)
            running_top_inds = merged_inds.gather(1, selected_pos)
        if running_top_vals is None:
            running_top_vals = torch.empty(len(q_batch), 0)
            running_top_inds = torch.empty(len(q_batch), 0, dtype=torch.long)
        all_scores.extend(running_top_vals.tolist())
        all_indices.extend(running_top_inds.tolist())
    save_results(output_dir, raw_queries, all_indices, all_scores, raw_corpus)
    print("Done.")

if __name__ == "__main__":
    main()
