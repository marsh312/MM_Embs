#!/usr/bin/env bash
set -e

# Activate environment
source /root/anaconda3/bin/activate /share/project/liuze/Emb_Eval

echo "Processing source: annualreports"
CUDA_VISIBLE_DEVICES=5 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/E5-Mistral/eval_e5_contextual.py --source annualreports
echo "Finished source: annualreports"
echo "----------------------------------------"

echo "Processing source: arxiv_cs"
CUDA_VISIBLE_DEVICES=5 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/E5-Mistral/eval_e5_contextual.py --source arxiv_cs
echo "Finished source: arxiv_cs"
echo "----------------------------------------"

echo "Processing source: arxiv_math"
CUDA_VISIBLE_DEVICES=5 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/E5-Mistral/eval_e5_contextual.py --source arxiv_math
echo "Finished source: arxiv_math"
echo "----------------------------------------"

echo "Processing source: arxiv_phy"
CUDA_VISIBLE_DEVICES=5 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/E5-Mistral/eval_e5_contextual.py --source arxiv_phy
echo "Finished source: arxiv_phy"
echo "----------------------------------------"

echo "Processing source: chemrxiv"
CUDA_VISIBLE_DEVICES=5 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/E5-Mistral/eval_e5_contextual.py --source chemrxiv
echo "Finished source: chemrxiv"
echo "----------------------------------------"

echo "Processing source: courtlistener"
CUDA_VISIBLE_DEVICES=5 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/E5-Mistral/eval_e5_contextual.py --source courtlistener
echo "Finished source: courtlistener"
echo "----------------------------------------"

echo "Processing source: pmc"
CUDA_VISIBLE_DEVICES=5 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/E5-Mistral/eval_e5_contextual.py --source pmc
echo "Finished source: pmc"
echo "----------------------------------------"

echo "Processing source: worldbank"
CUDA_VISIBLE_DEVICES=5 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/E5-Mistral/eval_e5_contextual.py --source worldbank
echo "Finished source: worldbank"
echo "----------------------------------------"
