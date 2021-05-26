import argparse
from datetime import datetime
import json
import re
import random
import shutil
from pathlib import Path

import yaml

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

SLURM_HEADER = """#!/bin/bash
#SBATCH --array=0-{n_jobs}%50
#SBATCH --job-name=coherence
#SBATCH --output={log_dir}/coherence-%A-%a.log
#SBATCH --constraint=cpu-large
#SBATCH --cpus-per-task=16
"""

def load_json(path):
    with open(path) as infile:
        return json.load(infile)


def load_tokens(path):
    """
    Stream tokens from a file.
    """
    with open(path) as infile:
        for text in infile:
            text = text.strip()
            if text:
                yield text.split(" ")


def save_json(obj, path):
    with open(path, 'w') as outfile:
        return json.dump(obj, outfile, indent=2)


def save_text(obj, path):
    with open(path, "w") as outfile:
        outfile.write(obj)


def load_yaml(path):
    with open(path) as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def gen_measure_name(coherence_measure, window_size, reference_corpus, top_n):
    """
    Make a unique measure name from the arguments
    """
    window_size = f"_{window_size}" if window_size else ""
    measure_name = f"{coherence_measure}{window_size}_{reference_corpus}"
    if top_n != 10:
        measure_name += f"_top{top_n}"
    return measure_name


def collect_topics(topic_dir, start_at=0, eval_every_n=1, eval_last_only=False):
    """Get all topics from the directory"""
    paths = [
        (int(re.search("[0-9]+", str(path.name)).group()), path)
        for path in Path(topic_dir).glob("*[0-9].txt")
    ]
    paths = sorted(paths, key=lambda x: x[0])

    if eval_last_only:
        idx, path = paths[-1]
        return [(idx, path, list(load_tokens(path)))]

    return [
        (idx, path, list(load_tokens(path)))
        for idx, path in paths 
        if idx >= start_at and idx % eval_every_n == 0
    ]


def make_dictionary(data_dir, cleanups=None):
    """
    Create a dictionary in the input directory to save processing time in the future
    """
    data_dict = Dictionary(load_tokens(Path(data_dir, "train.txt")))
    data_dict.save(str(Path(data_dir, "train-dict.npy")))
    return data_dict


def backup_coherences(input_dir):
    """
    Consolidate and backup the existing coherence files in a directory
    """
    coherence_paths = Path(args.input_dir).glob("**/coherences.json")
    coherences = {str(p): load_json(p) for p in coherence_paths}
    backup_path = Path(args.input_dir, f"{datetime.now():%Y%m%d}-coherences.json")
    if backup_path.exists():
        backup_path = Path(args.input_dir, f"{datetime.now():%Y%m%d_%H%M%S}-coherences.json")
    save_json(coherences, backup_path)
    print(f"Found and backed up {len(coherences)} coherence.json files at {backup_path}")
    return coherences


def make_runs(args, save=True):
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

    measure_name = gen_measure_name(
        args.coherence_measure, args.window_size, args.reference_corpus, args.top_n
    )

    commands = []
    for topic_dir in Path(args.input_dir).glob("**/topics"):
        topic_dir = topic_dir.absolute()

        if not args.update_existing and (topic_dir.parent / "coherences.json").exists():
            coh = load_json(topic_dir.parent / "coherences.json")
            if measure_name in coh:
                continue

        if not list(topic_dir.glob("*.txt")):
            continue
        commands.append(cmd_template.format(input_dir=topic_dir))
    if not save:
        return commands

    slurm_log_dir = Path(args.input_dir, "_run-logs/coherence/slurm-logs")
    slurm_log_dir.mkdir(exist_ok=True, parents=True)

    slurm_header = SLURM_HEADER.format(n_jobs=len(commands)-1, log_dir=slurm_log_dir)
    commands = [slurm_header] + [
        f"test ${{SLURM_ARRAY_TASK_ID}} -eq {run_id} && {run_command}"
        for run_id, run_command in enumerate(commands)
    ]
    slurm_sbatch_script = "\n".join(commands)
    print(f"found {len(commands)} runs")
    save_text(slurm_sbatch_script, Path(slurm_log_dir.parent, f"{datetime.now():%Y%m%d_%H%M%S}-coherence-runs.sh"))
    return slurm_sbatch_script


def calculate_coherence(args):
    topic_dir = Path(args.input_dir)
    parent_dir = topic_dir.parent
    config = load_yaml(parent_dir / "config.yml")
    data_dir = config["input_dir"]

    #### quick HACK to handle scratch directories, needs cleanup ###
    processed_name = Path(data_dir).name
    data_dir_map = {
        f"/workspace/topic-preprocessing/data/nytimes/processed/{processed_name}": f"/scratch/{processed_name}/nytimes",
        f"/workspace/topic-preprocessing/data/wikitext/processed/{processed_name}": f"/scratch/{processed_name}/wikitext",
        f"/workspace/topic-preprocessing/data/bbc/processed/{processed_name}": f"/scratch/{processed_name}/bbc",
    }

    # for out-of-sample coherence
    ref_corpus = args.reference_corpus
    if ref_corpus == "wikitext_full" or ref_corpus == "nytimes_full":
        try:
            data_dict = Dictionary.load(str(Path(data_dir, "train-dict.npy")))
        except FileNotFoundError:
            data_dict = make_dictionary(data_dir)
    
        if ref_corpus == "wikitext_full":
            mapped_dir = f"/workspace/topic-preprocessing/data/wikitext/processed/{processed_name}"
        if ref_corpus == "nytimes_full":
            mapped_dir = f"/workspace/topic-preprocessing/data/nytimes/processed/{processed_name}"
        ref_corpus_fname = "full.txt"
    # standard coherence
    else:
        ref_corpus = args.reference_corpus
        ref_corpus_fname = f"{ref_corpus}.txt" # can later update to external if needed
        mapped_dir = Path(data_dir_map[data_dir])

        if Path(mapped_dir, "train-dict.npy").exists() and Path(mapped_dir, ref_corpus_fname).exists():
            print("reading files from scratch", flush=True)
            data_dict = Dictionary.load(str(Path(mapped_dir, "train-dict.npy")))
        else: 
    else: 
        else: 
            print("loading files", flush=True)
            try:
                data_dict = Dictionary.load(str(Path(data_dir, "train-dict.npy")))
            except FileNotFoundError:
                data_dict = make_dictionary(data_dir)

            # copy to scratch directory
            print("copying files to scratch", flush=True)
            mapped_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(Path(data_dir, ref_corpus_fname), Path(mapped_dir, ref_corpus_fname))
            shutil.copy(Path(data_dir, "train-dict.npy"), Path(mapped_dir, "train-dict.npy"))

    ### end hack ###

    topic_sets = collect_topics(
        topic_dir=topic_dir,
        start_at=args.start_at,
        eval_every_n=args.eval_every_n,
        eval_last_only=args.eval_last_only,
    )

    measure_name = gen_measure_name(
        args.coherence_measure, args.window_size, args.reference_corpus, args.top_n
    )
    coherence_results = {measure_name: {}}

    print("calculating coherence...", flush=True)
    for idx, path, topics in topic_sets:
        topics = [t[:args.top_n] for t in topics]
        reference_text = load_tokens(Path(mapped_dir, ref_corpus_fname))
        
        cm = CoherenceModel(
            topics=topics,
            texts=reference_text,
            dictionary=data_dict,
            coherence=args.coherence_measure,
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
    if output_dir.exists(): # TODO: currently broken, will overwrite different epochs
        prev_coherence = load_json(output_dir)
        prev_coherence.update(**coherence_results)
        coherence_results = prev_coherence

    save_json(coherence_results, parent_dir / "coherences.json")
    print("done!")
    return coherence_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["create_runs", "calculate_coherence", "make_dictionary", "backup"])
    parser.add_argument("--input_dir")
    parser.add_argument("--start_at", type=int)
    parser.add_argument("--eval_every_n", type=int)
    parser.add_argument("--eval_last_only", action="store_true", default=False)
    parser.add_argument("--coherence_measure", default="c_v", choices=['u_mass', 'c_v', 'c_uci', 'c_npmi'])
    parser.add_argument("--reference_corpus", default="test", choices=["val", "train", "test", "full", "wikitext_full", "nytimes_full"])
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--python_path", default="/workspace/.conda/envs/gensim/bin/python")
    parser.add_argument("--update_existing", action="store_true", default=False, help="Update existing measures")

    args = parser.parse_args()

    if args.mode == "make_dictionary":
        dictionary = make_dictionary(args.input_dir)
    if args.mode == "backup":
        coherences = backup_coherences(args.input_dir)
    if args.mode == "create_runs":
        slurm_sbatch_script = make_runs(args)
        save_text(slurm_sbatch_script, "coherence-runs.sh")
    elif args.mode == "calculate_coherence":
        calculate_coherence(args)
    

    