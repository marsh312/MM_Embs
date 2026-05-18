import os
import json
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Fix missing qa_type using another model directory.")
    parser.add_argument("--src", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260311-2/Target-AnsCtx_Corpus-Page-TargetPageOnly/Qwen3-Embedding",
                        help="Source model dir (Qwen3-Embedding, missing qa_type)")
    parser.add_argument("--ref", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260311-2/Target-AnsCtx_Corpus-Page-TargetPageOnly/BGE-M3",
                        help="Reference model dir (BGE-M3, correct qa_type)")
    parser.add_argument("--out", type=str, default="/share/project/liuze/projects/contextual_mmemb/data/generated_QAs/testdatas_results/20260311-2/Target-AnsCtx_Corpus-Page-TargetPageOnly/fix-Qwen3-Embedding",
                        help="Output directory for fixed files")
    return parser.parse_args()


def build_qid_map(ref_dir):
    """
    Build q_id -> qa_type mapping from reference directory
    """
    qid_map = {}

    for root, _, files in os.walk(ref_dir):
        for file in files:
            if file.endswith(".jsonl"):
                path = os.path.join(root, file)

                with open(path, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            qid = data.get("q_id")
                            qa_type = data.get("qa_type")

                            if qid and qa_type:
                                qid_map[qid] = qa_type
                        except:
                            continue

    print(f"Loaded {len(qid_map)} q_id mappings from reference.")
    return qid_map


def fix_files(src_dir, out_dir, qid_map):
    """
    Fix qa_type in src_dir and write to out_dir
    """
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".jsonl"):
                src_path = os.path.join(root, file)

                # 保持目录结构
                rel_path = os.path.relpath(root, src_dir)
                out_folder = os.path.join(out_dir, rel_path)
                os.makedirs(out_folder, exist_ok=True)

                out_path = os.path.join(out_folder, file)

                total = 0
                fixed = 0
                missing = 0

                with open(src_path, "r") as fin, open(out_path, "w") as fout:
                    for line in fin:
                        total += 1
                        try:
                            data = json.loads(line)
                            qid = data.get("q_id")

                            if qid in qid_map:
                                data["qa_type"] = qid_map[qid]
                                fixed += 1
                            else:
                                data["qa_type"] = "unknown"
                                missing += 1

                            fout.write(json.dumps(data) + "\n")

                        except:
                            continue

                print(f"{src_path} → fixed: {fixed}/{total}, missing: {missing}")


def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Building q_id → qa_type map...")
    qid_map = build_qid_map(args.ref)

    print("Fixing files...")
    fix_files(args.src, args.out, qid_map)

    print("Done!")


if __name__ == "__main__":
    main()