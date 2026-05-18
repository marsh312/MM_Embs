#!/usr/bin/env bash
set -euo pipefail

EVAL_PY="/share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-VL/eval_bge_vl.py"
MODEL_PATH="BAAI/BGE-VL-base"

benchmarks=(
  "FinRAGBench-V"
  "LongDocURL"
  "MMLongBench-Doc"
  "SlideVQA"
  "UniDoc-Bench/commerce_manufacturing"
  "UniDoc-Bench/construction"
  "UniDoc-Bench/crm"
  "UniDoc-Bench/education"
  "UniDoc-Bench/energy"
  "UniDoc-Bench/finance"
  "UniDoc-Bench/healthcare"
  "UniDoc-Bench/legal"
  "ViDoRe_v3/computer_science"
  "ViDoRe_v3/energy"
  "ViDoRe_v3/finance_en"
  "ViDoRe_v3/finance_fr"
  "ViDoRe_v3/hr"
  "ViDoRe_v3/industrial"
  "ViDoRe_v3/pharmaceuticals"
  "ViDoRe_v3/physics"
)

# 轮询均分：偶数下标 -> wave1，奇数下标 -> wave2（最终各 10 个）
wave1=()
wave2=()
for i in "${!benchmarks[@]}"; do
  if (( i % 2 == 0 )); then
    wave1+=("${benchmarks[$i]}")
  else
    wave2+=("${benchmarks[$i]}")
  fi
done

echo "Wave1 (${#wave1[@]}): ${wave1[*]}"
echo "Wave2 (${#wave2[@]}): ${wave2[*]}"

# -------------------------
# Example Usage
# -------------------------
# CUDA_VISIBLE_DEVICES=0 python "$EVAL_PY" --model_path "$MODEL_PATH" --score_method joint   --tasks "${wave1[@]}" &
# CUDA_VISIBLE_DEVICES=1 python "$EVAL_PY" --model_path "$MODEL_PATH" --score_method image   --tasks "${wave1[@]}" &
# CUDA_VISIBLE_DEVICES=2 python "$EVAL_PY" --model_path "$MODEL_PATH" --score_method context --tasks "${wave1[@]}" &
# CUDA_VISIBLE_DEVICES=3 python "$EVAL_PY" --model_path "$MODEL_PATH" --score_method fusion  --tasks "${wave1[@]}" &
