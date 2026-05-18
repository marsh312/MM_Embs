import os
import json
import torch
import numpy as np
from argparse import ArgumentParser
import datasets
from tqdm import tqdm

from colpali_dataset import ColPaliQueryDataset, ColPaliCorpusDataset
from colpali_model import ColPaliEmbed

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
    parser.add_argument("--model_path", type=str, default="/share/project/shared_models/colpali-v1.3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--base_data_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Paths
    base_data_dir = args.base_data_dir
    query_path = os.path.join(base_data_dir, args.source, "query.jsonl")
    corpus_path = os.path.join(base_data_dir, args.source, "corpus.jsonl")
    
    output_dir = os.path.join(base_data_dir.replace("/testdatas", "/testdatas_results"), "ColPali", args.source)
    
    # 1. Init Model
    model = ColPaliEmbed(model_path=args.model_path)
    
    print(f"Processing {args.source}...")
    print(f"Query Path: {query_path}")
    print(f"Corpus Path: {corpus_path}")
    print(f"Output Dir: {output_dir}")

    try:
        # Load queries
        with open(query_path, 'r') as f:
            raw_queries = [json.loads(line) for line in f]
        
        raw_corpus = datasets.load_dataset('json', data_files=corpus_path, split='train')

        query_ds = ColPaliQueryDataset(raw_queries)
        corpus_ds = ColPaliCorpusDataset(raw_corpus)

        # 2. Encode Queries (Text)
        print("Encoding Queries...")
        # Note: ColPaliEmbed.get_query_embeddings expects list of strings
        q_embs = model.get_query_embeddings(
            texts=[item['text'] for item in query_ds], 
            batch_size=args.batch_size
        )

        # 3. Encode Corpus (Image)
        print("Encoding Corpus...")
        # Note: ColPaliEmbed.get_image_embeddings expects list of image paths/objects
        c_embs = model.get_image_embeddings(
            images=[item['image'] for item in corpus_ds],
            batch_size=args.batch_size
        )

        # 4. Retrieval (MaxSim / ColBERT style)
        print("Calculating Scores...")
        all_indices = []
        all_scores = []
        
        score_batch_size = 100 # Queries per batch
        
        for i in tqdm(range(0, len(q_embs), score_batch_size)):
            q_batch = q_embs[i : i + score_batch_size]
            
            # Move q_batch to GPU
            q_batch_gpu = [q.to(model.device) for q in q_batch]
            
            batch_scores = []
            c_chunk_size = 100 # Docs per chunk
            
            for j in range(0, len(c_embs), c_chunk_size):
                c_chunk = c_embs[j : j + c_chunk_size]
                c_chunk_gpu = [c.to(model.device) for c in c_chunk]
                
                with torch.inference_mode():
                    # Check if score_multi_vector expects lists or stacked tensors
                    # ColPaliProcessor.score_multi_vector signature:
                    # (qs: List[torch.Tensor], ds: List[torch.Tensor], batch_size: int = 128, device: str = 'cpu') -> torch.Tensor
                    # It returns [n_queries, n_docs]
                    # But here we are processing in chunks manually.
                    # We pass the GPU tensors directly.
                    # Wait, if we pass GPU tensors, we should probably check if score_multi_vector handles it.
                    # The user snippet: scores = processor.score_multi_vector(query_embeddings, image_embeddings)
                    # where query_embeddings is output of model().
                    
                    scores_chunk = model.processor.score_multi_vector(q_batch_gpu, c_chunk_gpu)
                    # scores_chunk is [B_q, B_c] tensor
                
                batch_scores.append(scores_chunk.cpu())
                
                del c_chunk_gpu
                torch.cuda.empty_cache()
            
            # Concatenate scores for this query batch along doc dimension
            if batch_scores:
                final_batch_scores = torch.cat(batch_scores, dim=1) # [B_q, Total_Docs]
            else:
                final_batch_scores = torch.empty(len(q_batch), 0)

            # Top-k
            top_k = min(args.k, final_batch_scores.shape[1])
            vals, inds = torch.topk(final_batch_scores, top_k, dim=1)
            
            all_scores.extend(vals.tolist())
            all_indices.extend(inds.tolist())
            
            del q_batch_gpu

        # 5. Save
        save_results(output_dir, raw_queries, all_indices, all_scores, raw_corpus)
        print("Done.")

    except Exception as e:
        print(f"Error processing {args.source}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
