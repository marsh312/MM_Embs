source /root/anaconda3/bin/activate /share/project/liuze/Envs/Qwen3VL_Emb


# python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py --source arxiv_cs
export CUDA_VISIBLE_DEVICES=0
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py --source arxiv_math
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py --source arxiv_phy
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py --source chemrxiv
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py --source courtlistener
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py --source pmc
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py --source worldbank
python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Qwen3VL-Emb/eval_qwen3vl_contextual.py --source annualreports