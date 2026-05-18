import os
import json
import argparse
import numpy as np
from collections import defaultdict
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260311-2/Target-AnsCtx_Corpus-Page-TargetPageOnly")
    # parser.add_argument("--base_dir", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260331_multi_5_20_text_drop_0.05_0.2/Target-AnsCtx_Corpus-Page-TargetPageOnly")
    parser.add_argument("--target_model", type=str, default="BM25")
    parser.add_argument("--out_dir", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260331_Qwen3_emb_multi_5_BM25_multi_5_protect_5_text_1/Target-AnsCtx_Corpus-Page-TargetPageOnly")
    parser.add_argument("--good_rank", type=int, default=10)
    parser.add_argument("--bad_rank", type=int, default=20)
    return parser.parse_args()





def get_rank(data):
    cur_ranks = data.get("cur_ranks", {})
    if not cur_ranks:
        return None
    return list(cur_ranks.values())[0]


def load_model_ranks(base_dir, models):
    all_ranks = {}

    for m in models:
        print(f"Loading {m}...")
        qid_to_rank = {}

        model_dir = os.path.join(base_dir, m)

        for domain in os.listdir(model_dir):
            domain_dir = os.path.join(model_dir, domain)
            if not os.path.isdir(domain_dir):
                continue

            file_path = os.path.join(domain_dir, "retrieval_output.jsonl")
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        qid = data["q_id"]
                        rank = get_rank(data)
                        if rank is not None:
                            qid_to_rank[qid] = rank
                    except:
                        continue

        all_ranks[m] = qid_to_rank

    return all_ranks


def main():
    args = parse_args()
    random.seed(42)

    ignore_models = {"Qwen3-Embedding"}

    models = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
        and d not in ignore_models
    ])

    assert args.target_model in models, "target model not found"

    print("Loading all model ranks...")
    all_ranks = load_model_ranks(args.base_dir, models)
    target_ranks = all_ranks[args.target_model]

    # ===== STEP 1: 决定哪些 q_id 保留 =====
    print("\nSelecting valid q_ids...")

    keep_qids = set()

    type_stats_before = defaultdict(int)
    type_stats_after = defaultdict(int)

    total = 0
    removed = 0

    src_dir = os.path.join(args.base_dir, args.target_model)

    for domain in os.listdir(src_dir):
        domain_dir = os.path.join(src_dir, domain)
        if not os.path.isdir(domain_dir):
            continue

        file_path = os.path.join(domain_dir, "retrieval_output.jsonl")
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except:
                    continue

                total += 1

                qid = data["q_id"]
                qa_type = data.get("qa_type", "unknown")

                type_stats_before[qa_type] += 1

                # ===== multimodal 全保留 =====
                if qa_type in ["section", "multihop"]:
                    # keep_qids.add(qid)
                    # type_stats_after[qa_type] += 1
                    # continue
                    # for model in ["BM25", "fix-Qwen3-Embedding"]:
                    model = "BM25"

                    rank = all_ranks[model].get(qid, 1000)
                    if rank <= 5:
                        if random.random() > 0.05:
                            continue
                    elif rank <= 20:
                        if random.random() > 0.2:
                            continue
                    # elif rank <= 50:
                    #     if random.random() > 0.5:
                    #         continue
                        # >20 全保留
                    model = "fix-Qwen3-Embedding"
                    rank = all_ranks[model].get(qid, 1000)
                    if rank <= 5:
                        # if random.random() > 0.05:
                        continue
                    elif rank <= 20:
                        if random.random() > 0.1:
                            continue
                    # elif rank <= 50:
                    #     if random.random() > 0.5:
                    #         continue
                # ===== section / multihop 筛选 =====
                if qa_type in ["multimodal"]:
                    bm25_rank = all_ranks["BM25"].get(qid, 1000)
                    fix_rank = all_ranks["fix-Qwen3-Embedding"].get(qid, 1000)
                    nemotron_rank = all_ranks["Nemotron-ColEmbed_VL-4B-V2"].get(qid, 1000)              

                    # ===== 去掉弱模型 =====
                    valid_models = [m for m in models if m not in ["E5-Mistral", "BM25", "fix-Qwen3-Embedding"]]             

                    ranks = [all_ranks[m].get(qid, 1000) for m in valid_models]
                    mean_rank = np.mean(ranks)              

                    # ===== CASE 1: easy（必须保留）=====
                    # if median_rank <= 20:
                    #     keep_qids.add(qid)
                    #     type_stats_after[qa_type] += 1
                    #     continue                

                    # ===== CASE 2: 多模态优势 =====
                    # if nemotron_rank + 10 < bm25_rank:
                    #     keep_qids.add(qid)
                    #     type_stats_after[qa_type] += 1
                    #     continue                
                    if mean_rank <= 5:
                        keep_qids.add(qid)
                        type_stats_after[qa_type] += 1
                        continue
                    # ===== CASE 3: BM25 独强 =====
                    if bm25_rank <= 6:
                        if random.random() > 0.05:
                            removed += 1
                            continue                

                    # ===== CASE 4: fix 独强 =====
                    if fix_rank <= 5:
                        if random.random() > 0.05:
                            removed += 1
                            continue                

                keep_qids.add(qid)
                type_stats_after[qa_type] += 1

                # keep_qids.add(qid)
                # type_stats_after[qa_type] += 1

    print(f"Selected {len(keep_qids)} / {total} q_ids")

    # ===== STEP 2: 对所有模型写文件 =====
    print("\nWriting filtered dataset for ALL models...")

    kept = 0

    for model in models:
        model_dir = os.path.join(args.base_dir, model)

        for domain in os.listdir(model_dir):
            domain_dir = os.path.join(model_dir, domain)
            if not os.path.isdir(domain_dir):
                continue

            in_file = os.path.join(domain_dir, "retrieval_output.jsonl")
            if not os.path.exists(in_file):
                continue

            out_domain_dir = os.path.join(args.out_dir, model, domain)
            os.makedirs(out_domain_dir, exist_ok=True)

            out_file = os.path.join(out_domain_dir, "retrieval_output.jsonl")

            with open(in_file, "r") as fin, open(out_file, "w") as fout:
                for line in fin:
                    try:
                        data = json.loads(line)
                        qid = data["q_id"]
                    except:
                        continue

                    if qid in keep_qids:
                        fout.write(json.dumps(data) + "\n")
                        kept += 1

    # ===== stats =====
    print("\n===== DATASET STATS =====")
    print(f"Original: {total}")
    print(f"Kept QIDs: {len(keep_qids)}")
    print(f"Removed QIDs: {removed}")

    print("\n===== QA_TYPE BEFORE =====")
    for k, v in type_stats_before.items():
        print(f"{k:12s}: {v}")

    print("\n===== QA_TYPE AFTER =====")
    for k, v in type_stats_after.items():
        print(f"{k:12s}: {v}")

    # 保存 stats
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_qids": total,
            "kept_qids": len(keep_qids),
            "removed_qids": removed,
            "before": dict(type_stats_before),
            "after": dict(type_stats_after)
        }, f, indent=2)


if __name__ == "__main__":
    main()