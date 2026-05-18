#!/bin/bash
set -e

# Default GPU IDs
GPU_IDS=${1:-0,1,2,3,4,5,6,7}

echo "Starting evaluation run with GPUs: $GPU_IDS"

BASE_DIR="/share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers"

# List of retriever directories
RETRIEVERS=(
    "BGE-M3"
    "BGE-VL"
    "BM25"
    "ColQwen"
    "E5-Mistral"
    "ColPali"
    "DSE-Qwen2"
    "GME"
    "Qwen3-Embedding"
    "Qwen3VL-Emb"
    "RzenEmbed"
    "Tomoro-ColQwen3"
)

for RETRIEVER in "${RETRIEVERS[@]}"; do
    SCRIPT_PATH="$BASE_DIR/$RETRIEVER/bash_run_eval_2026_03_11.sh"
    
    if [ -f "$SCRIPT_PATH" ]; then
        echo "========================================"
        echo "Running evaluation for $RETRIEVER..."
        echo "Script: $SCRIPT_PATH"
        echo "GPUs: $GPU_IDS"
        echo "========================================"
        
        # Run the script with the GPU IDs
        bash "$SCRIPT_PATH" "$GPU_IDS"
        
        echo "Finished evaluation for $RETRIEVER"
        echo ""
    else
        echo "Warning: Script not found for $RETRIEVER at $SCRIPT_PATH"
    fi
done

echo "All evaluations completed."
