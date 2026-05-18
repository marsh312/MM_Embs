import os
import json
import pandas as pd
from tabulate import tabulate

def main():
    base_path = "/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results"
    retrievers = [
        "BM25", 
        "E5-Mistral",
        "BGE-M3", 
        "Qwen3-Embedding", 
        "GME",
        "BGE-VL",
        "RzenEmb", 
        "Qwen3VL-Emb", 
        "Tomoro-ColQwen3"
    ]
    
    # Collect all datasets first to ensure consistent columns
    all_datasets = set()
    results = {} # structure: {retriever: {dataset: score}}
    
    metric = "NDCG@10"
    
    for retriever in retrievers:
        retriever_path = os.path.join(base_path, retriever)
        results[retriever] = {}
        
        if not os.path.exists(retriever_path):
            print(f"Warning: Path not found for {retriever}")
            continue
            
        # List subdirectories (datasets)
        try:
            items = os.listdir(retriever_path)
        except OSError:
            continue
            
        for item in items:
            dataset_path = os.path.join(retriever_path, item)
            if os.path.isdir(dataset_path):
                result_file = os.path.join(dataset_path, "retrieval_results.json")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            
                        # Handle list if necessary (based on print_result.py logic)
                        if isinstance(data, list) and len(data) > 0:
                            data = data[0]
                        
                        if isinstance(data, dict):
                            score = data.get(metric)
                            if score is not None:
                                results[retriever][item] = float(score) * 100 # usually percentage
                                all_datasets.add(item)
                    except Exception as e:
                        print(f"Error reading {result_file}: {e}")

    # Create DataFrame
    sorted_datasets = sorted(list(all_datasets))
    
    # Build rows
    rows = []
    for retriever in retrievers:
        row = [retriever]
        for ds in sorted_datasets:
            val = results[retriever].get(ds, None)
            row.append(f"{val:.2f}" if val is not None else "-")
        rows.append(row)
        
    headers = ["Retriever"] + sorted_datasets
    
    print(tabulate(rows, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    main()
