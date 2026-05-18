#!/usr/bin/env bash
set -euo pipefail
source /root/anaconda3/bin/activate /share/project/liuze/Emb_Eval
# Path to the evaluation script
EVAL_PY="/share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3.py"

# Data paths
CORPUS_PATH="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas/arxiv_cs/corpus.jsonl"
QUERY_PATH="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas/arxiv_cs/query.jsonl"
PAGES_PATH="/share/project/liuze/projects/contextual_mmemb/data/pdf_datas/arxiv_cs/rawdata/pages.jsonl"

# Output directory

OUTPUT_DIR="/share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/results/arxiv_cs"

# Model path
MODEL_PATH="/share/project/shared_models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/"

echo "Starting BGE-M3 Evaluation on Arxiv CS..."

CUDA_VISIBLE_DEVICES=3 python "$EVAL_PY" \
    --corpus_path "$CORPUS_PATH" \
    --query_path "$QUERY_PATH" \
    --pages_path "$PAGES_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --k 200

echo "Evaluation completed. Results saved to $OUTPUT_DIR"
