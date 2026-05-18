source /root/anaconda3/bin/activate /share/project/liuze/Envs/MMEmbs/ColPali

# Set default model path if not provided
MODEL_PATH="/share/project/shared_models/colqwen2.5-v0.2"

# Run for different datasets
CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/ColQwen/eval_colqwen_contextual.py --source courtlistener --model_path $MODEL_PATH

# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/ColQwen/eval_colqwen_contextual.py --source arxiv_cs --model_path $MODEL_PATH
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/ColQwen/eval_colqwen_contextual.py --source arxiv_math --model_path $MODEL_PATH
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/ColQwen/eval_colqwen_contextual.py --source arxiv_phy --model_path $MODEL_PATH
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/ColQwen/eval_colqwen_contextual.py --source chemrxiv --model_path $MODEL_PATH
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/ColQwen/eval_colqwen_contextual.py --source pmc --model_path $MODEL_PATH
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/ColQwen/eval_colqwen_contextual.py --source worldbank --model_path $MODEL_PATH
# CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/ColQwen/eval_colqwen_contextual.py --source annualreports --model_path $MODEL_PATH
