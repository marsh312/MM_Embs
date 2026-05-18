import json

file_path = "/share/project/liuze/projects/contextual_mmemb/data/pdf_datas/arxiv_cs/rawdata/table_figure_pairs.jsonl"
target_doc_id = "1609.00893"
output_file = "/share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/1609.00893_data_all.json"

found_data = []

print(f"Searching for doc_id: {target_doc_id} in {file_path}...")

with open(file_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("doc_id") == target_doc_id:
                found_data.append(data)
        except json.JSONDecodeError:
            continue

print(f"Found {len(found_data)} records.")

with open(output_file, 'w') as out_f:
    json.dump(found_data, out_f, indent=2, ensure_ascii=False)

print(f"All data saved to {output_file}")
