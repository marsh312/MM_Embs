#!/bin/bash

# 显存优化
source /root/anaconda3/bin/activate /share/project/junjie/env/reasonir

# 数据源列表
SOURCES=("arxiv_cs" )


# 循环遍历每个数据源并运行评估脚本
for source in "${SOURCES[@]}"; do
    echo "Processing source: $source"
    CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bgevl_contextual.py \
        --source "$source" \
        --batch_size 16 \
        --k 100
    echo "Finished source: $source"
    echo "----------------------------------------"
done


CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bgevl_contextual.py --source arxiv_math --batch_size 16 --k 100
CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bgevl_contextual.py --source arxiv_phy --batch_size 16 --k 100
CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bgevl_contextual.py --source chemrxiv --batch_size 16 --k 100
CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bgevl_contextual.py --source courtlistener --batch_size 16 --k 100
CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bgevl_contextual.py --source pmc --batch_size 16 --k 100
CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bgevl_contextual.py --source worldbank --batch_size 16 --k 100
CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bgevl_contextual.py --source annualreports --batch_size 16 --k 100