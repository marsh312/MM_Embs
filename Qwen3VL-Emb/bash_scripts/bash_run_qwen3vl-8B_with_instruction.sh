#!/usr/bin/env bash
set -euo pipefail
source /root/anaconda3/bin/activate /share/project/liuze/Envs/Qwen3VL_Emb
EVAL_PY="/share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_with_instruction.py"

benchmarks=(
  "ViDoRe_v3/computer_science"
  "ViDoRe_v3/energy"
  "ViDoRe_v3/finance_en"
  "ViDoRe_v3/finance_fr"
  "ViDoRe_v3/hr"
  "ViDoRe_v3/industrial"
  "ViDoRe_v3/pharmaceuticals"
  "ViDoRe_v3/physics"
  # "FinRAGBench-V"
  # "LongDocURL"
  # "MMLongBench-Doc"
  # "SlideVQA"
  # "UniDoc-Bench/commerce_manufacturing"
  # "UniDoc-Bench/construction"
  # "UniDoc-Bench/crm"
  # "UniDoc-Bench/education"
  # "UniDoc-Bench/energy"
  # "UniDoc-Bench/finance"
  # "UniDoc-Bench/healthcare"
  # "UniDoc-Bench/legal"
)

# 轮询均分：偶数下标 -> wave1，奇数下标 -> wave2（最终各 4 个）
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
# Wave 1: GPU 0-3
# -------------------------
CUDA_VISIBLE_DEVICES=0 python "$EVAL_PY" --score_method joint   --tasks "${wave1[@]}" --model_path /share/project/shared_models/Qwen3-VL-Embedding-8B --output_root /share/project/liuze/projects/contextual_mmemb/results/Qwen3VL-Emb-8B_with-instruction --batch_size 32 &
CUDA_VISIBLE_DEVICES=1 python "$EVAL_PY" --score_method image   --tasks "${wave1[@]}" --model_path /share/project/shared_models/Qwen3-VL-Embedding-8B --output_root /share/project/liuze/projects/contextual_mmemb/results/Qwen3VL-Emb-8B_with-instruction --batch_size 32 &
CUDA_VISIBLE_DEVICES=2 python "$EVAL_PY" --score_method context --tasks "${wave1[@]}" --model_path /share/project/shared_models/Qwen3-VL-Embedding-8B --output_root /share/project/liuze/projects/contextual_mmemb/results/Qwen3VL-Emb-8B_with-instruction --batch_size 32 &
CUDA_VISIBLE_DEVICES=3 python "$EVAL_PY" --score_method fusion  --tasks "${wave1[@]}" --model_path /share/project/shared_models/Qwen3-VL-Embedding-8B --output_root /share/project/liuze/projects/contextual_mmemb/results/Qwen3VL-Emb-8B_with-instruction --batch_size 32 &

# -------------------------
# Wave 2: GPU 4-7
# -------------------------
CUDA_VISIBLE_DEVICES=4 python "$EVAL_PY" --score_method joint   --tasks "${wave2[@]}" --model_path /share/project/shared_models/Qwen3-VL-Embedding-8B --output_root /share/project/liuze/projects/contextual_mmemb/results/Qwen3VL-Emb-8B_with-instruction --batch_size 32 &
CUDA_VISIBLE_DEVICES=5 python "$EVAL_PY" --score_method image   --tasks "${wave2[@]}" --model_path /share/project/shared_models/Qwen3-VL-Embedding-8B --output_root /share/project/liuze/projects/contextual_mmemb/results/Qwen3VL-Emb-8B_with-instruction --batch_size 32 &
CUDA_VISIBLE_DEVICES=6 python "$EVAL_PY" --score_method context --tasks "${wave2[@]}" --model_path /share/project/shared_models/Qwen3-VL-Embedding-8B --output_root /share/project/liuze/projects/contextual_mmemb/results/Qwen3VL-Emb-8B_with-instruction --batch_size 32 &
CUDA_VISIBLE_DEVICES=7 python "$EVAL_PY" --score_method fusion  --tasks "${wave2[@]}" --model_path /share/project/shared_models/Qwen3-VL-Embedding-8B --output_root /share/project/liuze/projects/contextual_mmemb/results/Qwen3VL-Emb-8B_with-instruction --batch_size 32 &

wait
echo "All done."
