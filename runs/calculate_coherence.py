import argparse
from datetime import datetime
import json
import re
import shutil
from pathlib import Path

import yaml

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

SLURM_HEADER = """#!/bin/bash
#SBATCH --array=0-{n_jobs}%50
#SBATCH --job-name=coherence
#SBATCH --output={log_dir}/coherence-%A-%a.log
#SBATCH --constraint=cpu-med
#SBATCH --exclusive
"""

def load_json(path):
    with open(path) as infile:
        return json.load(infile)


def load_tokens(path):
    with open(path) as infile:
        return [text.strip().split(" ") for text in infile]


def save_json(obj, path):
    with open(path, 'w') as outfile:
        return json.dump(obj, outfile, indent=2)


def save_text(obj, path):
    with open(path, "w") as outfile:
        outfile.write(obj)


def load_yaml(path):
    with open(path) as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def collect_topics(topic_dir, start_at=0, eval_every_n=1, eval_last_only=False):
    """Get all topics from the directory"""
    paths = [
        (int(re.search("[0-9]+", str(path)).group()), path)
        for path in Path(topic_dir).glob("*[0-9].txt")
    ]
    paths = sorted(paths, key=lambda x: x[0])
    if eval_last_only:
        idx, path = paths[-1]
        return [(idx, path, load_tokens(path))]

    return [
        (idx, path, load_tokens(path))
        for idx, path in paths 
        if idx >= start_at and idx % eval_every_n == 0
    ]


def make_runs(args):
    """
    Generate slurm runs to calculate coherence en masse
    """
    cmd_template = (
        f"{args.python_path} {__file__} "
        "--mode calculate_coherence "
        "--input_dir {input_dir} "
        f"--coherence_measure {args.coherence_measure} "
        f"--top_n {args.top_n} "
        f"--reference_corpus {args.reference_corpus} "
    )
    if args.window_size:
        cmd_template += f" --window_size {args.window_size}"
    if args.start_at:
        cmd_template += f" --start_at {args.start_at}"
    if args.eval_every_n:
        cmd_template += f" --eval_every_n {args.eval_every_n}"
    if args.eval_last_only:
        cmd_template += f" --eval_last_only"

    commands = []
    for topic_dir in Path(args.input_dir).glob("**/topics"):
        topic_dir = topic_dir.absolute()
        if not list(topic_dir.glob("*.txt")):
            continue
        if "/dvae/" in str(topic_dir):
            continue
        commands.append(cmd_template.format(input_dir=topic_dir))

    slurm_log_dir = Path(args.input_dir, "_run-logs/coherence/slurm-logs")
    slurm_log_dir.mkdir(exist_ok=True, parents=True)

    slurm_header = SLURM_HEADER.format(n_jobs=len(commands)-1, log_dir=slurm_log_dir)
    commands = [slurm_header] + [
        f"test ${{SLURM_ARRAY_TASK_ID}} -eq {run_id} && {run_command}"
        for run_id, run_command in enumerate(commands)
    ]
    slurm_sbatch_script = "\n".join(commands)
    save_text(slurm_sbatch_script, Path(slurm_log_dir.parent, f"{datetime.now():%Y%m%d_%H%M%S}-coherence-runs.sh"))
    return slurm_sbatch_script


def calculate_coherence(args):
    topic_dir = Path(args.input_dir)
    parent_dir = topic_dir.parent
    config = load_yaml(parent_dir / "config.yml")
    data_dir = config["input_dir"]

    #### quick hack to handle scratch directories ###
    data_dir_map = {
        "/workspace/topic-preprocessing/data/nytimes/processed/vocab_25k-mindf_0.0001_or_3-maxdf_0.9": "/scratch/nytimes",
        "/workspace/topic-preprocessing/data/wikitext/processed/vocab_25k-mindf_0.0001_or_3-maxdf_0.9": "/scratch/wikitext",
        "/workspace/topic-preprocessing/data/bbc/processed/vocab_25k-mindf_0.0001_or_3-maxdf_0.9": "/scratch/bbc",
    }
    
    mapped_dir = Path(data_dir_map[data_dir])
    # HACK: BBC is too small for good estimates
    ref_corpus = "train" if "bbc" in str(data_dir) else args.reference_corpus
    ref_corpus_fname = f"{ref_corpus}.txt" # can later update to Wikipedia or whatever

    if Path(mapped_dir, "train-dict.npy").exists() and Path(mapped_dir, ref_corpus_fname).exists():
        print("reading files from scratch")
        data_dict = Dictionary.load(str(Path(mapped_dir, "train-dict.npy")))
        reference_text = load_tokens(Path(mapped_dir, ref_corpus_fname))
    else: 
        data_dict = Dictionary(load_tokens(Path(data_dir, "train.txt")))
        reference_text = load_tokens(Path(data_dir, ref_corpus_fname))

        # copy to scratch directory
        print("copying files to scratch")
        mapped_dir.mkdir(exist_ok=True)
        shutil.copy(Path(data_dir, ref_corpus_fname), Path(mapped_dir, ref_corpus_fname))
        data_dict.save(str(Path(mapped_dir, "train-dict.npy")))
    ### end hack ###

    topic_sets = collect_topics(
        topic_dir=topic_dir,
        start_at=args.start_at,
        eval_every_n=args.eval_every_n,
        eval_last_only=args.eval_last_only,
    )
    
    measure = args.coherence_measure
    win_size = f"_{args.window_size}" if args.window_size else ""

    measure_name = f"{measure}{win_size}_{ref_corpus}"
    coherence_results = {measure_name: {}}

    print("calculating coherence...")
    for idx, path, topics in topic_sets:
        topics = [t[:args.top_n] for t in topics]
        cm = CoherenceModel(
            topics=topics,
            texts=reference_text,
            dictionary=data_dict,
            coherence=measure,
            window_size=args.window_size,
        )
        confirmed_measures = cm.get_coherence_per_topic()
        mean = cm.aggregate_measures(confirmed_measures)
        coherence_results[measure_name][idx] = {
            "aggregate": float(mean),
            "by_topic": [float(i) for i in confirmed_measures], # needs to be python float to json-serialize
            "path": str(path),
        }
    output_dir = parent_dir / "coherences.json"
    if output_dir.exists():
        prev_coherence = load_json(output_dir)
        prev_coherence.update(**coherence_results)
        coherence_results = prev_coherence

    save_json(coherence_results, parent_dir / "coherences.json")
    return coherence_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["create_runs", "calculate_coherence"])
    parser.add_argument("--input_dir")
    parser.add_argument("--start_at", type=int)
    parser.add_argument("--eval_every_n", type=int)
    parser.add_argument("--eval_last_only", action="store_true", default=False)
    parser.add_argument("--coherence_measure", default="c_v", choices=['u_mass', 'c_v', 'c_uci', 'c_npmi'])
    parser.add_argument("--reference_corpus", default="val", choices=["val", "train"])
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--python_path", default="/workspace/.conda/envs/gensim/bin/python")

    args = parser.parse_args()

    if args.mode == "create_runs":
        slurm_sbatch_script = make_runs(args)
        save_text(slurm_sbatch_script, "coherence-runs.sh")
    elif args.mode == "calculate_coherence":
        calculate_coherence(args)
    

    