source /root/anaconda3/bin/activate /share/project/liuze/Envs/Qwen3VL_Emb


CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/DSE-Qwen2/eval_dse_contextual.py --source courtlistener

# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/DSE-Qwen2/eval_dse_contextual.py --source arxiv_cs
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/DSE-Qwen2/eval_dse_contextual.py --source arxiv_math
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/DSE-Qwen2/eval_dse_contextual.py --source arxiv_phy
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/DSE-Qwen2/eval_dse_contextual.py --source chemrxiv
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/DSE-Qwen2/eval_dse_contextual.py --source pmc
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/DSE-Qwen2/eval_dse_contextual.py --source worldbank
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/DSE-Qwen2/eval_dse_contextual.py --source annualreports
