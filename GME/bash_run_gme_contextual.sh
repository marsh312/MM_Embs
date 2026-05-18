source /root/anaconda3/bin/activate /share/project/junjie/env/reasonir



CUDA_VISIBLE_DEVICES=2 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/GME/eval_gme_contextual.py --source arxiv_cs
CUDA_VISIBLE_DEVICES=2 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/GME/eval_gme_contextual.py --source arxiv_math
CUDA_VISIBLE_DEVICES=2 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/GME/eval_gme_contextual.py --source arxiv_phy
CUDA_VISIBLE_DEVICES=2 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/GME/eval_gme_contextual.py --source chemrxiv
CUDA_VISIBLE_DEVICES=2 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/GME/eval_gme_contextual.py --source courtlistener
CUDA_VISIBLE_DEVICES=2 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/GME/eval_gme_contextual.py --source pmc
CUDA_VISIBLE_DEVICES=2 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/GME/eval_gme_contextual.py --source worldbank
CUDA_VISIBLE_DEVICES=2 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/GME/eval_gme_contextual.py --source annualreports