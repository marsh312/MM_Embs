#!/bin/bash
# Environment
source /root/anaconda3/bin/activate /share/project/liuze/Envs/Qwen3VL_Emb

# Default GPU IDs
GPU_IDS=${1:-4,5,6,7}
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "Using GPUs: ${GPUS[@]}"

BASE_DIRS=(
"/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas/20260311-2/Target-Ans_Corpus-Page-TargetPageOnly"
"/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas/20260311-2/Target-AnsCtx_Corpus-Page-TargetPageOnly"
)

SOURCES=(
"annualreports"
"arxiv_cs"
"arxiv_math"
"arxiv_phy"
"chemrxiv"
"courtlistener"
"pmc"
"worldbank"
)

SCRIPT_PATH="/share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py"

for BASE_DIR in "${BASE_DIRS[@]}"; do
    echo "Processing Base Dir: $BASE_DIR"
    
    pids=()
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_ID=${GPUS[i]}
        (
            for ((j=i; j<${#SOURCES[@]}; j+=NUM_GPUS)); do
                SOURCE=${SOURCES[j]}
                echo "Running $SOURCE on GPU $GPU_ID"
                CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_PATH" --source "$SOURCE" --base_data_dir "$BASE_DIR"
            done
        ) &
        pids+=($!)
    done
    
    # Wait for all GPU workers
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo "Finished Base Dir: $BASE_DIR"
done
