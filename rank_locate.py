import os
import json
import argparse
from collections import defaultdict

import os
import json
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260331_Qwen3_emb_multi_50_10_BM25_multi_50_10/Target-AnsCtx_Corpus-Page-TargetPageOnly")
    return parser.parse_args()


def get_rank(data):
    cur_ranks = data.get("cur_ranks", {})
    if not cur_ranks:
        return None
    return list(cur_ranks.values())[0]


def bucket(rank):
    if rank is None:
        return "miss"
    if rank <= 1:
        return "≤1"
    elif rank <= 3:
        return "≤3"
    elif rank <= 5:
        return "≤5"
    elif rank <= 10:
        return "≤10"
    elif rank <= 20:
        return "≤20"
    else:
        return ">20"


def main():
    args = parse_args()

    models = sorted([d for d in os.listdir(args.base_dir)
                     if os.path.isdir(os.path.join(args.base_dir, d))])

    buckets = ["≤1", "≤3", "≤5", "≤10", "≤20", ">20", "miss"]

    # model → bucket
    stats = defaultdict(lambda: defaultdict(int))

    # model → domain → bucket
    stats_domain = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # model → qa_type → bucket
    stats_qatype = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    avg_rank = defaultdict(list)

    for model in models:
        model_dir = os.path.join(args.base_dir, model)

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
                    except:
                        continue

                    rank = get_rank(data)
                    b = bucket(rank)
                    qa_type = data.get("qa_type", "unknown")

                    # overall
                    stats[model][b] += 1

                    # domain
                    stats_domain[model][domain][b] += 1

                    # qa_type
                    stats_qatype[model][qa_type][b] += 1

                    if rank is not None:
                        avg_rank[model].append(rank)

    # ===== 1. overall =====
    print("\n===== OVERALL =====")
    header = ["Model"] + buckets + ["MeanRank"]
    print("".join(h.ljust(10) for h in header))

    for m in models:
        row = [m]
        total = sum(stats[m][b] for b in buckets)

        for b in buckets:
            cnt = stats[m][b]
            ratio = cnt / total * 100 if total > 0 else 0
            row.append(f"{ratio:.1f}%")
            

        mean = sum(avg_rank[m]) / len(avg_rank[m]) if avg_rank[m] else 0
        row.append(f"{mean:.2f}")

        print("".join(str(x).ljust(10) for x in row))

    # ===== 2. per domain =====
    print("\n===== PER DOMAIN =====")
    for m in models:
        print(f"\n### MODEL: {m}")
        for d in stats_domain[m]:
            print(f"\n[{d}]")

            total = sum(stats_domain[m][d][b] for b in buckets)

            for b in buckets:
                cnt = stats_domain[m][d][b]
                ratio = cnt / total * 100 if total > 0 else 0
                print(f"{b:5s}: {cnt:5d} ({ratio:.1f}%)")

    # ===== 3. per qa_type =====
    print("\n===== PER QA_TYPE (TABLE) =====")

    buckets = ["≤1", "≤3", "≤5", "≤10", "≤20", ">20", "miss"]

    # 收集所有 qa_type
    all_qatypes = set()
    for m in stats_qatype:
        all_qatypes.update(stats_qatype[m].keys())

    all_qatypes = sorted(list(all_qatypes))

    for qt in all_qatypes:
        print(f"\n===== QA_TYPE: {qt} =====")

        # 表头
        header = ["Model"] + buckets + ["MeanRank"]
        col_widths = [12] + [8] * (len(buckets) + 1)

        # 打印表头
        row = ""
        for i, h in enumerate(header):
            row += h.ljust(col_widths[i])
        print(row)
        print("-" * sum(col_widths))

        # 每个模型一行
        for m in models:
            row = ""
            row += m.ljust(col_widths[0])

            total = sum(stats_qatype[m][qt][b] for b in buckets)

            for i, b in enumerate(buckets):
                cnt = stats_qatype[m][qt][b]
                ratio = cnt / total * 100 if total > 0 else 0
                row += f"{ratio:.1f}%".ljust(col_widths[i + 1])

            # mean rank（只统计这个 qa_type）
            ranks = []
            for domain in stats_domain[m]:
                # 重新读一遍不划算，这里简单用 overall avg 近似
                pass

            # 简化：不算 per-qa_type mean（如果你要我可以加）
            row += "-".ljust(col_widths[-1])

            print(row)


if __name__ == "__main__":
    main()
