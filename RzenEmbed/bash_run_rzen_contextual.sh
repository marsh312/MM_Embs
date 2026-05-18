#!/usr/bin/env bash
set -euo pipefail

# 更稳的激活方式
source /root/anaconda3/bin/activate /share/project/liuze/Envs/MMEmb

# 基本信息
which python
python -V
python -c "import sys; print('executable=', sys.executable)"

# 打开 Python 崩溃回溯（对很多 C 扩展崩溃有帮助）
export PYTHONFAULTHANDLER=1

# 线程相关先收敛，避免 MKL/OpenMP 类 segfault
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u \
  /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/RzenEmbed/eval_rzen_contextual.py \
  --source arxiv_cs


CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/RzenEmbed/eval_rzen_contextual.py --source arxiv_math
CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/RzenEmbed/eval_rzen_contextual.py --source arxiv_phy
CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/RzenEmbed/eval_rzen_contextual.py --source chemrxiv
CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/RzenEmbed/eval_rzen_contextual.py --source courtlistener
CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/RzenEmbed/eval_rzen_contextual.py --source pmc
CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/RzenEmbed/eval_rzen_contextual.py --source worldbank
CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/RzenEmbed/eval_rzen_contextual.py --source annualreports