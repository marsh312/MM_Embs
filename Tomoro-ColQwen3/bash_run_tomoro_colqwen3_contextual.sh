source /root/anaconda3/bin/activate /share/project/liuze/Envs/MMEmb


CUDA_VISIBLE_DEVICES=4 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Tomoro-ColQwen3/eval_tomoro_colqwen3_contextual.py --source arxiv_cs

CUDA_VISIBLE_DEVICES=4 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Tomoro-ColQwen3/eval_tomoro_colqwen3_contextual.py --source arxiv_math
CUDA_VISIBLE_DEVICES=4 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Tomoro-ColQwen3/eval_tomoro_colqwen3_contextual.py --source arxiv_phy
CUDA_VISIBLE_DEVICES=4 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Tomoro-ColQwen3/eval_tomoro_colqwen3_contextual.py --source chemrxiv
CUDA_VISIBLE_DEVICES=4 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Tomoro-ColQwen3/eval_tomoro_colqwen3_contextual.py --source courtlistener
CUDA_VISIBLE_DEVICES=4 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Tomoro-ColQwen3/eval_tomoro_colqwen3_contextual.py --source pmc
CUDA_VISIBLE_DEVICES=4 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Tomoro-ColQwen3/eval_tomoro_colqwen3_contextual.py --source worldbank
CUDA_VISIBLE_DEVICES=4 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Tomoro-ColQwen3/eval_tomoro_colqwen3_contextual.py --source annualreports