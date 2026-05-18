#!/usr/bin/env bash
set -e

# Activate environment
source /root/anaconda3/bin/activate /share/project/liuze/Emb_Eval

echo "Processing source: annualreports"
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BM25/eval_bm25_contextual.py --source annualreports
echo "Finished source: annualreports"
echo "----------------------------------------"

echo "Processing source: arxiv_cs"
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BM25/eval_bm25_contextual.py --source arxiv_cs
echo "Finished source: arxiv_cs"
echo "----------------------------------------"

echo "Processing source: arxiv_math"
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BM25/eval_bm25_contextual.py --source arxiv_math
echo "Finished source: arxiv_math"
echo "----------------------------------------"

echo "Processing source: arxiv_phy"
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BM25/eval_bm25_contextual.py --source arxiv_phy
echo "Finished source: arxiv_phy"
echo "----------------------------------------"

echo "Processing source: chemrxiv"
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BM25/eval_bm25_contextual.py --source chemrxiv
echo "Finished source: chemrxiv"
echo "----------------------------------------"

echo "Processing source: courtlistener"
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BM25/eval_bm25_contextual.py --source courtlistener
echo "Finished source: courtlistener"
echo "----------------------------------------"

echo "Processing source: pmc"
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BM25/eval_bm25_contextual.py --source pmc
echo "Finished source: pmc"
echo "----------------------------------------"

echo "Processing source: worldbank"
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BM25/eval_bm25_contextual.py --source worldbank
echo "Finished source: worldbank"
echo "----------------------------------------"
