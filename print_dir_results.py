import json
import os
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Print retrieval benchmark results for models in a directory.")
    parser.add_argument("--path", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260311-2/Target-AnsCtx_Corpus-Page-TargetPageOnly", help="Base path containing model directories.")
    parser.add_argument("--metric", type=str, default="NDCG@10", help="The metric key (e.g., NDCG@10).")
    return parser.parse_args()

def load_results(base_path, metric_key):
    """Load results from the directory structure."""
    results = defaultdict(dict)
    all_datasets = set()
    models = []

    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist.")
        return results, [], []

    # Get all subdirectories in base_path as potential models
    try:
        entries = os.listdir(base_path)
        for entry in entries:
            full_path = os.path.join(base_path, entry)
            if os.path.isdir(full_path):
                models.append(entry)
    except Exception as e:
        print(f"Error reading directory {base_path}: {e}")
        return results, [], []

    models.sort()  # Sort models alphabetically

    for model in models:
        model_path = os.path.join(base_path, model)
        
        # Walk through the model directory to find retrieval_results.json
        for root, dirs, files in os.walk(model_path):
            if "retrieval_results.json" in files:
                file_path = os.path.join(root, "retrieval_results.json")
                
                # Determine dataset name relative to model_path
                rel_path = os.path.relpath(root, model_path)
                dataset_name = rel_path.replace("\\", "/") # Unify path separators
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Handle case where JSON might be a list or dict
                        if isinstance(data, list) and len(data) > 0:
                            data = data[0]
                        
                        if metric_key in data:
                            val = float(data[metric_key])
                            results[model][dataset_name] = val
                            all_datasets.add(dataset_name)
                except Exception as e:
                    # print(f"Warning: Failed to read {file_path}: {e}")
                    pass

    return results, sorted(list(all_datasets)), models

def main():
    args = parse_args()
    results, all_datasets, models = load_results(args.path, args.metric)

    if not models:
        print("No models found.")
        return

    if not all_datasets:
        print("No datasets found.")
        return

    # Prepare table data
    # Header: Model | Dataset1 | Dataset2 | ... | Mean
    header = ["Model"] + all_datasets + ["Mean"]
    table_rows = [header]

    for model in models:
        row = [model]
        vals = []
        for dataset in all_datasets:
            if dataset in results[model]:
                val = results[model][dataset]
                vals.append(val)
                row.append(f"{val * 100:.2f}")
            else:
                row.append("-")
        
        if vals:
            mean_val = sum(vals) / len(vals)
            row.append(f"{mean_val * 100:.2f}")
        else:
            row.append("-")
        
        table_rows.append(row)

    # Print table with manual formatting
    print(f"\nMetric: {args.metric}")
    print(f"Base Path: {args.path}")
    print("-" * 30)

    # Calculate column widths
    col_widths = []
    for i in range(len(header)):
        max_w = max(len(str(r[i])) for r in table_rows)
        col_widths.append(max_w + 2)

    # Print rows
    for i, row in enumerate(table_rows):
        formatted = []
        for c_idx, cell in enumerate(row):
            w = col_widths[c_idx]
            # First column left-aligned, others right-aligned
            if c_idx == 0:
                formatted.append(str(cell).ljust(w))
            else:
                formatted.append(str(cell).rjust(w))
        
        print("".join(formatted))
        if i == 0:
            print("-" * sum(col_widths))

if __name__ == "__main__":
    main()
