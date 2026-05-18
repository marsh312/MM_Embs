source /root/anaconda3/bin/activate /share/project/liuze/Envs/MMEmbs/Nemotron_ColEmbed_Qwen

# CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Nemotron-ColEmbed/eval_nemotron_colembed_contextual.py --source courtlistener  --batch_size 32

# source /root/anaconda3/bin/activate /share/project/liuze/Envs/MMEmbs/Nemotron_ColEmbed_Qwen_torch_2.8

# CUDA_VISIBLE_DEVICES=7 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Nemotron-ColEmbed/eval_nemotron_colembed_contextual-new-env.py --source courtlistener --batch_size 32

cd /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Nemotron-ColEmbed
out_dir="/share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Nemotron-ColEmbed"
run_ts="$(date +%Y%m%d_%H%M%S)"
log_file="${out_dir}/log/arxiv_cs_${run_ts}.log"
timing_log_file="${out_dir}/log/arxiv_cs_${run_ts}.timing.jsonl"
echo "Logging to $log_file"
echo "Timing events to $timing_log_file"

CUDA_VISIBLE_DEVICES=1 python /share/project/liuze/projects/contextual_mmemb/code/baseline_retrievers/Nemotron-ColEmbed/eval_nemotron_colembed_contextual-new-env-new.py \
  --source arxiv_cs \
  --batch_size 32 \
  --timing_log_path "$timing_log_file" \
  2>&1 | tee -a "$log_file"
