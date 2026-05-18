import json
import os
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Print retrieval benchmark results with interleaved averages.")
    parser.add_argument("--metric", type=str, default="NDCG@10", help="The metric key (e.g., NDCG@10).")
    parser.add_argument("--path", type=str, default="/share/project/liuze/projects/contextual_mmemb/results/RzenEmbed", help="Base path results.")
    parser.add_argument("--only_complete_result", action="store_true", help="Only show datasets that have results for ALL input modes.")
    parser.add_argument("--filter_cols", type=str, default=None, help="Filter columns by a keyword and show their average.")
    return parser.parse_args()

def load_results(base_path, metric_key):
    """加载数据"""
    results = defaultdict(dict)
    all_datasets = set()
    
    # 固定的行顺序
    input_modes = [
        "context",
        "image", 
        "image+context_score",
        "image_score+context_score"
    ]

    for mode in input_modes:
        mode_path = os.path.join(base_path, mode)
        if not os.path.exists(mode_path):
            continue
        
        for root, dirs, files in os.walk(mode_path):
            if "retrieval_results.json" in files:
                file_path = os.path.join(root, "retrieval_results.json")
                rel_path = os.path.relpath(root, mode_path)
                dataset_name = rel_path.replace("\\", "/") # 统一路径格式
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            data = data[0]
                        
                        if metric_key in data:
                            results[mode][dataset_name] = float(data[metric_key])
                            all_datasets.add(dataset_name)
                except Exception:
                    pass

    return results, sorted(list(all_datasets)), input_modes

def filter_incomplete_datasets(results, all_datasets, input_modes):
    """过滤不完整的数据列"""
    complete_datasets = []
    for ds in all_datasets:
        is_complete = True
        for mode in input_modes:
            if ds not in results[mode]:
                is_complete = False
                break
        if is_complete:
            complete_datasets.append(ds)
    return complete_datasets

def calculate_hierarchical_avg(row_data, all_valid_datasets):
    """
    计算分层平均值，返回: (Total_Avg, {UniDoc-Avg: val, ViDoRe-Avg: val})
    """
    groups = {
        "UniDoc": [],
        "ViDoRe": [],
        "Others": []
    }

    # 1. 分组收集数值
    for ds in all_valid_datasets:
        val = row_data.get(ds)
        if val is None: continue

        if "UniDoc-Bench" in ds:
            groups["UniDoc"].append(val)
        elif "ViDoRe_v3" in ds:
            groups["ViDoRe"].append(val)
        else:
            groups["Others"].append(val)

    # 2. 计算组内平均
    group_means = []
    display_avgs = {"UniDoc-Avg": None, "ViDoRe-Avg": None}

    if groups["UniDoc"]:
        m = sum(groups["UniDoc"]) / len(groups["UniDoc"])
        group_means.append(m)
        display_avgs["UniDoc-Avg"] = m

    if groups["ViDoRe"]:
        m = sum(groups["ViDoRe"]) / len(groups["ViDoRe"])
        group_means.append(m)
        display_avgs["ViDoRe-Avg"] = m

    # 其他独立数据集直接参与总平均
    group_means.extend(groups["Others"])

    # 3. 计算总平均 (Macro Average of Groups)
    final_avg = None
    if group_means:
        final_avg = sum(group_means) / len(group_means)

    return final_avg, display_avgs

def organize_columns(all_datasets):
    """
    将列进行分组排序，生成最终的打印顺序
    格式: [Other1, ..., UniDoc1, ..., UniDoc-Avg, ViDoRe1, ..., ViDoRe-Avg, Total-Avg]
    """
    unidoc_cols = sorted([d for d in all_datasets if "UniDoc-Bench" in d])
    vidore_cols = sorted([d for d in all_datasets if "ViDoRe_v3" in d])
    # 既不是 UniDoc 也不是 ViDoRe 的列
    other_cols = sorted([d for d in all_datasets if d not in unidoc_cols and d not in vidore_cols])

    final_order = []
    
    # 1. 添加其他列 (通常是 FinRAGBench, SlideVQA 等)
    final_order.extend(other_cols)
    
    # 2. 添加 UniDoc 列及其平均值
    if unidoc_cols:
        final_order.extend(unidoc_cols)
        final_order.append("UniDoc-Avg") # 占位符
        
    # 3. 添加 ViDoRe 列及其平均值
    if vidore_cols:
        final_order.extend(vidore_cols)
        final_order.append("ViDoRe-Avg") # 占位符
        
    # 4. 添加总平均
    final_order.append("Total-Avg") # 占位符
    
    return final_order

def main():
    args = parse_args()
    results, all_datasets, input_modes = load_results(args.path, args.metric)

    # 过滤列
    if args.only_complete_result:
        all_datasets = filter_incomplete_datasets(results, all_datasets, input_modes)
    
    if args.filter_cols:
        all_datasets = [d for d in all_datasets if args.filter_cols in d]
        if not all_datasets:
            print(f"No datasets found containing filter: {args.filter_cols}")
            return

        display_cols = sorted(all_datasets) + ["Average"]
        header = ["Input Mode"] + display_cols
        table_rows = [header]

        for mode in input_modes:
            row = [mode]
            vals = []
            for col in sorted(all_datasets):
                if col in results[mode]:
                    v = results[mode][col]
                    vals.append(v)
                    row.append(f"{v * 100:.2f}")
                else:
                    row.append("-")
            
            if vals:
                avg_val = sum(vals) / len(vals)
                row.append(f"{avg_val * 100:.2f}")
            else:
                row.append("-")
            table_rows.append(row)

    else:
        # === 关键修改：生成交错的列顺序 ===
        # 这里的 display_cols 包含了实际的数据集名称，以及 "UniDoc-Avg" 这样的特殊占位符
        display_cols = organize_columns(all_datasets)
        
        # 准备表头
        header = ["Input Mode"] + display_cols
        table_rows = [header]

        for mode in input_modes:
            row = [mode]
            
            # 先计算这一行的所有统计数据
            final_avg, group_avgs = calculate_hierarchical_avg(results[mode], all_datasets)
            
            # 根据排好的顺序填充数据
            for col_name in display_cols:
                if col_name == "UniDoc-Avg":
                    val = group_avgs.get("UniDoc-Avg")
                    row.append(f"{val * 100:.2f}" if val is not None else "-")
                
                elif col_name == "ViDoRe-Avg":
                    val = group_avgs.get("ViDoRe-Avg")
                    row.append(f"{val * 100:.2f}" if val is not None else "-")
                
                elif col_name == "Total-Avg":
                    row.append(f"{final_avg * 100:.2f}" if final_avg is not None else "-")
                
                else:
                    # 这是一个普通的数据集列
                    if col_name in results[mode]:
                        row.append(f"{results[mode][col_name] * 100:.2f}")
                    else:
                        row.append("-")
            
            table_rows.append(row)

    # 打印表格
    print(f"\nMetric: {args.metric}   Model: {args.path.split('/')[-1]}")
    if args.only_complete_result:
        print("(Showing ONLY complete columns)")
    print("-" * 30)

    col_widths = []
    for i in range(len(header)):
        max_w = max(len(str(r[i])) for r in table_rows)
        col_widths.append(max_w + 2)

    for i, row in enumerate(table_rows):
        formatted = []
        for c_idx, cell in enumerate(row):
            w = col_widths[c_idx]
            # 数据列右对齐，第一列左对齐
            if c_idx == 0:
                formatted.append(str(cell).ljust(w))
            else:
                formatted.append(str(cell).rjust(w))
        
        print("".join(formatted))
        if i == 0:
            print("-" * sum(col_widths))

if __name__ == "__main__":
    main()