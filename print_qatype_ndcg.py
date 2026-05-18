import json
import os
import argparse
from collections import defaultdict
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Compute NDCG grouped by qa_type.")
    parser.add_argument("--path", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260331_Qwen3_emb_multi_5_BM25_multi_5_protect_7_text/Target-AnsCtx_Corpus-Page-TargetPageOnly")
    # parser.add_argument("--path", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260311-2/Target-AnsCtx_Corpus-Page-TargetPageOnly")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def compute_ndcg(target_ids, retrieved_ids, k=10):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in target_ids:
            dcg += 1.0 / math.log2(i + 2)

    # ideal DCG
    ideal_hits = min(len(target_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def load_results(base_path, k):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_datasets = set()
    all_qatypes = set()
    models = []

    if not os.path.exists(base_path):
        print(f"Path not found: {base_path}")
        return results, [], [], []

    # models
    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path):
            models.append(entry)

    models.sort()

    for model in models:
        model_path = os.path.join(base_path, model)

        for root, _, files in os.walk(model_path):
            for file in files:
                if file.endswith(".jsonl"):
                    file_path = os.path.join(root, file)

                    dataset_name = os.path.relpath(root, model_path).replace("\\", "/")
                    all_datasets.add(dataset_name)

                    with open(file_path, "r") as f:
                        for line in f:
                            try:
                                data = json.loads(line)

                                qa_type = data.get("qa_type", "unknown")
                                all_qatypes.add(qa_type)

                                target_ids = set(data["target_ids"])
                                retrieved_ids = data["retrieved_ids"]

                                ndcg = compute_ndcg(target_ids, retrieved_ids, k)

                                results[qa_type][model][dataset_name].append(ndcg)

                            except:
                                continue

    return results, sorted(all_datasets), models, sorted(all_qatypes)


def print_table(results, datasets, models, title):
    print(f"\n===== {title} =====")

    header = ["Model"] + datasets + ["Mean"]
    table = [header]

    for model in models:
        row = [model]
        vals = []

        for d in datasets:
            if d in results[model]:
                scores = results[model][d]
                if scores:
                    val = sum(scores) / len(scores)
                    vals.append(val)
                    row.append(f"{val * 100:.2f}")
                else:
                    row.append("-")
            else:
                row.append("-")

        if vals:
            row.append(f"{(sum(vals)/len(vals))*100:.2f}")
        else:
            row.append("-")

        table.append(row)

    # format print
    col_widths = [max(len(str(r[i])) for r in table) + 2 for i in range(len(header))]

    for i, row in enumerate(table):
        line = ""
        for j, cell in enumerate(row):
            if j == 0:
                line += str(cell).ljust(col_widths[j])
            else:
                line += str(cell).rjust(col_widths[j])
        print(line)

        if i == 0:
            print("-" * sum(col_widths))


def main():
    args = parse_args()

    results, datasets, models, qa_types = load_results(args.path, args.k)

    if not models or not datasets:
        print("No valid data found.")
        return

    # overall（所有 qa_type 合并）
    overall = defaultdict(lambda: defaultdict(list))

    for qt in qa_types:
        for m in results[qt]:
            for d in results[qt][m]:
                overall[m][d].extend(results[qt][m][d])

    print_table(overall, datasets, models, f"Overall NDCG@{args.k}")

    # 每个 qa_type 单独一张表
    for qt in qa_types:
        print_table(results[qt], datasets, models, f"{qt} NDCG@{args.k}")


if __name__ == "__main__":
    main()