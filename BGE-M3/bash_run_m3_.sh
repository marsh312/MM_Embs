source /root/anaconda3/bin/activate /share/project/liuze/Emb_Eval

CUDA_VISIBLE_DEVICES=6 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3_contextual.py --source annualreports
CUDA_VISIBLE_DEVICES=6 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3_contextual.py --source arxiv_cs
CUDA_VISIBLE_DEVICES=6 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3_contextual.py --source arxiv_math
CUDA_VISIBLE_DEVICES=6 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3_contextual.py --source arxiv_phy
CUDA_VISIBLE_DEVICES=6 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3_contextual.py --source chemrxiv
CUDA_VISIBLE_DEVICES=6 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3_contextual.py --source courtlistener
CUDA_VISIBLE_DEVICES=6 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3_contextual.py --source pmc
CUDA_VISIBLE_DEVICES=6 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/BGE-M3/eval_bge_m3_contextual.py --source worldbank