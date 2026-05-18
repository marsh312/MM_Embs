import json
import os
import argparse
from collections import defaultdict
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Compute retrieval metrics grouped by qa_type.")
    parser.add_argument("--path", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260331_Qwen3_emb_multi_5_BM25_multi_5_protect_5_text_1/Target-AnsCtx_Corpus-Page-TargetPageOnly")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--metrics",
        type=str,
        default="ndcg",
        help="Comma-separated metrics to compute. Choices: ndcg,hit,recall,mrr,precision,map"
    )
    return parser.parse_args()


def compute_ndcg(target_ids, retrieved_ids, k=10):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in target_ids:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(target_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def compute_hit(target_ids, retrieved_ids, k=10):
    return 1.0 if any(doc_id in target_ids for doc_id in retrieved_ids[:k]) else 0.0


def compute_recall(target_ids, retrieved_ids, k=10):
    if not target_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in target_ids)
    return hits / len(target_ids)


def compute_precision(target_ids, retrieved_ids, k=10):
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    hits = sum(1 for doc_id in topk if doc_id in target_ids)
    return hits / len(topk)


def compute_mrr(target_ids, retrieved_ids, k=10):
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in target_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_map(target_ids, retrieved_ids, k=10):
    if not target_ids:
        return 0.0

    hits = 0
    ap = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in target_ids:
            hits += 1
            ap += hits / (i + 1)

    denom = min(len(target_ids), k)
    return ap / denom if denom > 0 else 0.0


def compute_metric(metric_name, target_ids, retrieved_ids, k=10):
    if metric_name == "ndcg":
        return compute_ndcg(target_ids, retrieved_ids, k)
    elif metric_name == "hit":
        return compute_hit(target_ids, retrieved_ids, k)
    elif metric_name == "recall":
        return compute_recall(target_ids, retrieved_ids, k)
    elif metric_name == "precision":
        return compute_precision(target_ids, retrieved_ids, k)
    elif metric_name == "mrr":
        return compute_mrr(target_ids, retrieved_ids, k)
    elif metric_name == "map":
        return compute_map(target_ids, retrieved_ids, k)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")


def load_results(base_path, k, metrics):
    metric_results = {
        metric: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for metric in metrics
    }

    qa_type_counts = defaultdict(lambda: defaultdict(int))
    all_datasets = set()
    all_qatypes = set()
    models = []

    if not os.path.exists(base_path):
        print(f"Path not found: {base_path}")
        return metric_results, qa_type_counts, [], [], []

    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path):
            models.append(entry)

    models.sort()

    if not models:
        return metric_results, qa_type_counts, [], [], []

    count_model = models[0]

    for model in models:
        model_path = os.path.join(base_path, model)

        for root, _, files in os.walk(model_path):
            for file in files:
                if not file.endswith(".jsonl"):
                    continue

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

                            for metric in metrics:
                                val = compute_metric(metric, target_ids, retrieved_ids, k)
                                metric_results[metric][qa_type][model][dataset_name].append(val)

                            if model == count_model:
                                qa_type_counts[dataset_name][qa_type] += 1

                        except Exception:
                            continue

    return metric_results, qa_type_counts, sorted(all_datasets), models, sorted(all_qatypes)


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
            row.append(f"{(sum(vals) / len(vals)) * 100:.2f}")
        else:
            row.append("-")

        table.append(row)

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


def print_count_table(qa_type_counts, datasets, qa_types, title="QA_TYPE Counts by Dataset"):
    print(f"\n===== {title} =====")

    header = ["QA_Type"] + datasets + ["Total"]
    table = [header]

    for qt in qa_types:
        row = [qt]
        total = 0
        for d in datasets:
            cnt = qa_type_counts[d].get(qt, 0)
            row.append(str(cnt))
            total += cnt
        row.append(str(total))
        table.append(row)

    total_row = ["Total"]
    grand_total = 0
    for d in datasets:
        s = sum(qa_type_counts[d].values())
        total_row.append(str(s))
        grand_total += s
    total_row.append(str(grand_total))
    table.append(total_row)

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
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]

    metric_results, qa_type_counts, datasets, models, qa_types = load_results(args.path, args.k, metrics)

    if not models or not datasets:
        print("No valid data found.")
        return

    print_count_table(qa_type_counts, datasets, qa_types)

    for metric in metrics:
        results = metric_results[metric]

        overall = defaultdict(lambda: defaultdict(list))
        for qt in qa_types:
            for m in results[qt]:
                for d in results[qt][m]:
                    overall[m][d].extend(results[qt][m][d])

        print_table(overall, datasets, models, f"Overall {metric.upper()}@{args.k}")

        for qt in qa_types:
            print_table(results[qt], datasets, models, f"{qt} {metric.upper()}@{args.k}")


if __name__ == "__main__":
    main()