source /root/anaconda3/bin/activate /share/project/liuze/Envs/MMEmbs/Nemotron_ColEmbed_Qwen


cd /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Nemotron-ColEmbed

CUDA_VISIBLE_DEVICES=3 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Nemotron-ColEmbed/eval_nemotron_colembed_contextual.py \
  --source arxiv_cs \
  --batch_size 32 