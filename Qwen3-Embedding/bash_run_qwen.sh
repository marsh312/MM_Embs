#!/usr/bin/env bash
set -euo pipefail

# Path to the evaluation script
source /root/anaconda3/bin/activate /share/project/liuze/Emb_Eval

CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3-Embedding/eval_qwen.py --source arxiv_cs
CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3-Embedding/eval_qwen.py --source arxiv_math
CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3-Embedding/eval_qwen.py --source arxiv_phy
CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3-Embedding/eval_qwen.py --source chemrxiv
CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3-Embedding/eval_qwen.py --source courtlistener
CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3-Embedding/eval_qwen.py --source pmc
CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3-Embedding/eval_qwen.py --source worldbank
CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3-Embedding/eval_qwen.py --source annualreports