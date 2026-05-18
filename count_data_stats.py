import os
import json
from tabulate import tabulate

def count_lines(file_path):
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
    except FileNotFoundError:
        return 0
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0
    return count

def main():
    base_path = "/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas"
    
    datasets = []
    try:
        items = os.listdir(base_path)
    except OSError as e:
        print(f"Error accessing directory: {e}")
        return

    stats = []
    
    for item in sorted(items):
        dataset_path = os.path.join(base_path, item)
        if os.path.isdir(dataset_path):
            query_path = os.path.join(dataset_path, "query.jsonl")
            corpus_path = os.path.join(dataset_path, "corpus.jsonl")
            
            query_count = count_lines(query_path)
            corpus_count = count_lines(corpus_path)
            
            stats.append([item, query_count, corpus_count])
            
    headers = ["Dataset", "Query Count", "Corpus Count"]
    print(tabulate(stats, headers=headers, tablefmt="github"))
    
    # Calculate totals
    total_query = sum(x[1] for x in stats)
    total_corpus = sum(x[2] for x in stats)
    print(f"\nTotal Queries: {total_query}")
    print(f"Total Corpus: {total_corpus}")

if __name__ == "__main__":
    main()
