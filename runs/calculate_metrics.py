from collections import defaultdict
import itertools
import json
import re
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder

from soup_nuts.metrics import (
    normalized_mutual_info_score, # TODO: adjusted?
    adjusted_rand_score,
    topic_dists_over_runs,
    purity,
    unique_doc_words_over_runs,
    unique_topic_words_over_runs,
)

MODELS = [
    "dvae", "mallet", "scholar", "scholar_kd", "ctm",
]

# maps the input directory to dataset attributes
DATASETS = {
    "/workspace/topic-preprocessing/data/bills/processed/labeled/vocab_5k": {
        "name": "bills",
        "vocab_size": 5,
        "label_names": ["topic", "subtopic"], # inside <prefix>.metadata.jsonl file
    },
    "/workspace/topic-preprocessing/data/bills/processed/labeled/vocab_15k": {
        "name": "bills",
        "vocab_size": 15,
        "label_names": ["topic", "subtopic"],
    },
    "/workspace/topic-preprocessing/data/wikitext/processed/labeled/vocab_5k": {
        "name": "wikitext",
        "vocab_size": 5,
        "label_names": ["supercategory", "category", "subcategory"],
    },
    "/workspace/topic-preprocessing/data/wikitext/processed/labeled/vocab_15k": {
        "name": "wikitext",
        "vocab_size": 15,
        "label_names": ["supercategory", "category", "subcategory"],
    },
}


NUM_TOPICS = [
    25, 50, 100, 200
]

TOP_N_WORDS = 25
# signature is `predicted_labels, true_labels`
CLUSTER_METRICS = {
    "adj_rand": adjusted_rand_score,
    "nmi": normalized_mutual_info_score,
    "purity": purity,
    "inv_purity": lambda pred, true: purity(true, pred),
}
# signature is `doc_topic_runs, topic_word_runs`
STABILITY_METRICS = {
    "doc_topic": {
        "unique_doc_words": lambda dt, tw: unique_doc_words_over_runs(dt, tw, top_n=TOP_N_WORDS, summarize=True),
        "unique_doc_words_hard": lambda dt, tw: unique_doc_words_over_runs(dt, tw, top_n=TOP_N_WORDS, hard_assignment=True, summarize=True),
        "doc_topic_dists": lambda dt, tw: topic_dists_over_runs(doc_topic_runs=dt, summarize=True, tqdm_kwargs={"leave": False, "desc": "Calculating doc-topic dists."}),
    },
    "topic_word": {
        "unique_topic_words": lambda dt, tw: unique_topic_words_over_runs(tw, top_n=TOP_N_WORDS, summarize=True),
        "topic_word_dists": lambda dt, tw: topic_dists_over_runs(topic_word_runs=tw, summarize=True, tqdm_kwargs={"leave": False, "desc": "Calculating topic-word dists"}),
    },
}

# run a comprehensive subset of the data
# TODO: generate programmatically
DEBUG_INDICES = (
    list(range(0, 10, 2)) +    # ('ctm', 'bills', 5, 25)
    list(range(161, 171, 2)) + # ('dvae', 'bills', 5, 25)
    list(range(402, 412, 2)) + # ('mallet', 'wikitext', 5, 25)
    list(range(604, 614, 2)) + # ('scholar', 'wikitext', 15, 25)
    list(range(775, 785, 2))   # ('scholar_kd', 'wikitext', 15, 50)
)

def load_yaml(path):
    with open(path, "r") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def load_mallet_doc_topic(fpath, skip_first=False):
    """Helper function to read plain-text-formatted doc-topics into a numpy array"""
    with open(fpath) as infile:
        if skip_first:
            next(infile)
        return np.array([
            [float(x) for x in line.strip().split("\t")[2:]]
            for line in infile
        ])


def _model_id_from_path(path, models=MODELS):
    """Get model name from the saved model path"""
    model_pattern = ")/|/(".join(models)
    model_pattern = f"/({model_pattern})/"
    try:
        return re.search(model_pattern, str(path)).group(0).strip("/")
    except AttributeError:
        return None


def _dataset_id_from_path(path, datasets=DATASETS):
    """Get dataset from the input dataset path"""
    dataset_pattern = ")|(^".join(datasets)
    dataset_pattern = f"(^{dataset_pattern})"
    try:
        return re.search(dataset_pattern, str(path)).group(0)
    except AttributeError:
        raise KeyError(f"Dataset {path} not found")


def _load_dataset_labels(datasets=DATASETS):
    """Retrieve & recode the labels for each dataset"""
    for data_dir in datasets:
        # initialize label data
        label_names = datasets[data_dir]["label_names"]
        labels = {name: {"train": [], "test": []} for name in label_names}

        # populate label data
        for split in ["train", "test"]:
            with open(Path(data_dir, f"{split}.metadata.jsonl")) as infile:
                for line in infile:
                    line_data = json.loads(line)
                    for name in label_names:
                        labels[name][split].append(line_data[name])
        
        # recode label data
        for name in label_names:
            label_map = LabelEncoder()
            train, test = labels[name].pop("train"), labels[name].pop("test")
            label_map.fit(train + test)
            labels[name]["train"] = label_map.transform(train)
            labels[name]["test"] = label_map.transform(test)
            labels[name]["_map"] = label_map.classes_

        datasets[data_dir]["labels"] = labels
        
    return datasets


def load_estimates(model_dir, model_type=None):
    """
    Get doc-topic, topic-word estimates for various models
    """
    model_dir = Path(model_dir)
    if model_type is None:
        model_type = _model_id_from_path(model_dir)

    # Load the text document-topic estimate as a numpy matrix
    if model_type in ["dvae", "ctm"]:
        topic_word = np.load(model_dir / "beta.npy")
        doc_topic_train = np.load(model_dir / "train.theta.npy")
        doc_topic_test = np.load(model_dir / "test.theta.npy")
    if model_type == "mallet":
        topic_word = np.load(model_dir / "beta.npy")
        doc_topic_train = load_mallet_doc_topic(model_dir / "doctopics.txt")
        doc_topic_test = load_mallet_doc_topic(model_dir / "test.theta.txt", skip_first=True)
    if model_type in ["scholar", "scholar_kd"]:
        topic_word = np.load(model_dir / "beta.npz")["beta"]
        doc_topic_train = np.load(model_dir / "train.theta.npy")
        doc_topic_test = np.load(model_dir / "test.theta.npy")

    # these models do not produce normalized outputs
    if model_type in ["scholar", "scholar_kd", "dvae"]:
        topic_word = softmax(topic_word, axis=1)

    return topic_word, doc_topic_train, doc_topic_test


def calculate_metrics(base_run_dir, config_file="config.yml", debug=False, as_dataframe=False):
    """
    Load all the saved estimates and calculate metrics.
    TODO: break up into functions?
    """
    # first, get and clean label data
    datasets = _load_dataset_labels(DATASETS)

    # collect all the configuration data
    config_fpaths = Path(base_run_dir).glob(f"**/{config_file}")
    run_data = []
    for fpath in tqdm(config_fpaths, desc="Loading configs"):
        config = load_yaml(fpath)

        dataset_id = _dataset_id_from_path(config["input_dir"])
        dataset = DATASETS[dataset_id]["name"]
        vocab_size = DATASETS[dataset_id]["vocab_size"]
        model_type = _model_id_from_path(fpath, models=MODELS)
        if model_type is None:
            continue
    
        data = {
            "model_dir": fpath.parent,
            "model_type": _model_id_from_path(fpath),
            "dataset": dataset,
            "vocab_size": vocab_size,
            "ds_id": dataset_id,
            "num_topics": config["num_topics"],
        }
        run_data.append(data)

    # now, loop through each group of settings to calculate metrics
    group_keys = ["model_type", "dataset", "vocab_size", "num_topics"]
    grouper = lambda c: tuple(c[k] for k in group_keys)
    run_data.sort(key=grouper)
    if debug:
        run_data = [run_data[i] for i in DEBUG_INDICES]
    total = len(set(grouper(c) for c in run_data))
    group_data = {} # initialize the aggregate group data

    with tqdm(total=total, desc="Calculating metrics") as pbar:
        for setting, run_group in itertools.groupby(run_data, key=grouper):
            # Initialize objects to store estimates and metrics
            pbar.set_postfix({k: s for k, s in zip(group_keys, setting)})
            estimates = defaultdict(list)

            # Load in the estimates
            for run in tqdm(run_group, desc="Loading estimates", leave=False):
                # beta: topic-words, theta: doc-topics
                beta, theta_train, theta_test = load_estimates(run['model_dir'], run['model_type'])
                # make assignments
                pred_train, pred_test = theta_train.argmax(1), theta_test.argmax(1)
                # if any identical assignments: pathological run, throw away
                if pred_train.max() == pred_train.min():
                    print("Pathological run found, skipping")
                    continue

                # calculate cluster metrics. `run` is a reference to item in `config_data`
                for metric_name, fn in CLUSTER_METRICS.items():
                    labels = datasets[run["ds_id"]]["labels"]
                    for label_name, data in labels.items():
                        run[f"{metric_name}_{label_name}_train"] = fn(pred_train, data["train"])
                        run[f"{metric_name}_{label_name}_test"] = fn(pred_test, data["test"])

                # save estimates for stability calculations
                estimates["beta"].append(beta)
                estimates["theta_train"].append(theta_train)
                estimates["theta_test"].append(theta_test)

            # calculate the stability metrics, over a given setting group
            stability_results = {}
            for metric_name, fn in STABILITY_METRICS["doc_topic"].items():
                stability_results[f"{metric_name}_train"] = fn(estimates["theta_train"], estimates["beta"])
                stability_results[f"{metric_name}_test"] = fn(estimates["theta_test"], estimates["beta"])
            for metric_name, fn in STABILITY_METRICS["topic_word"].items():
                stability_results[f"{metric_name}"] = fn(None, estimates["beta"])
            group_data[setting] = stability_results if not as_dataframe else pd.DataFrame(stability_results)
            pbar.update()

    if as_dataframe:
        run_data = pd.DataFrame(run_data)
        group_data = pd.concat(group_data, names=group_keys+["stat"]).reset_index()

    return run_data, group_data


def summarize_runs(run_data, group_keys, remove_inv_purity=True):
    """
    Find the optimal runs based on cluster quality metrics

    # TODO: geom mean?
    """
    # First, get averages over labels then over train/test
     
    summarized = run_data.groupby(group_keys).agg([np.mean, np.std])

        
if __name__ == "__main__":
    """
    This script calculates metrics for runs of various models.

    Since we need to run forward-passes of models that potentially rely on different, 
    possibly conflicting dependencies, this script must be run per-model (presumably 
    with different conda environments)

    Main idea: loop through each run in a directory, collect the estimates, and compute
    the metrics.

    Metric list:
        - Coverage
            - Purity
            - Rand index
            - NMI
            - Which topics have few assignments
        - Stability
            - Pairwise topic distances (perhaps some AUC-type-thing)
            - Pairwise reconstructed
            - 
            - ?
        - Extra
            - 
            - Diversity
            - Coherence

    IMPORTANT TODO:
        - Are the pairwise distances truly comparable between models, or are there systematic
        "architectural" differences (e.g., use of softmax) that may bias results?

        - Some runs will have 11 (not 10)---so total dists will be higher. Probably want mean of distance sums?
    
    TODO:
        X- Functions to retrieve estimates for each model with same signature/API
        X- Defer imports depending on the model
        - Updating existing runs
        - Writing out the "best" model & its corresponding configuration file (as yaml)
        to support future runs over seeds
        - Probably worth doing NPMI with wikipedia since we already have the plumbing
        X- Stability of test predictions also probably important
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--update",
        action="store_true",
        help="If the output file exists, update existing values", #TODO
    )
    args = parser.parse_args()

    # setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    cluster_fpath = output_dir / "cluster_metrics.csv"
    stability_fpath = output_dir / "stability_metrics.csv"

    if not args.overwrite and (cluster_fpath.exists() or stability_fpath.exists()):
        raise FileExistsError(f"Files exist in {args.output_dir} but `--overwrite` not used.")        

    run_df, group_df = calculate_metrics(args.base_run_dir, debug=args.debug, as_dataframe=True)

    # save the data
    run_df.to_csv(cluster_fpath, index=False)
    group_df.to_csv(stability_fpath, index=False)

    # TODO: get best setting per model
    
    run_df.groupby(["model_type", "dataset", "vocab_size"])

