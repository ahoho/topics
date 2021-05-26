from pathlib import Path
import json
import sys

import pandas as pd

def load_json(path):
    with open(path) as infile:
        return json.load(infile)


if __name__ == "__main__":
    DATA_DIR = sys.argv[1]
    DATA_FILE = sys.argv[2]
    fname = Path(DATA_DIR, DATA_FILE)
    raw_data = load_json(fname)

    auto_metric_names = [
        k for k in raw_data['wikitext']['etm']['metrics']
        if not (k.startswith("ratings") or k.startswith("intrusion") or k.endswith("top15") or k.endswith("top5"))
    ]
    auto_metric_names += [k.replace("nytimes", "wikitext") for k in auto_metric_names if "nytimes" in k]
    auto_metric_names = sorted(auto_metric_names)

    intrusion_rows = []
    ratings_rows = []

    n_ratings_annotators = len(raw_data['wikitext']['etm']['metrics']['ratings_scores_raw'][0])
    n_intrusion_annotators = len(raw_data['wikitext']['etm']['metrics']['intrusion_scores_raw'][0])
    n_topics = 50
    for dataset in raw_data:
        for model in raw_data[dataset]:
            metric_data = raw_data[dataset][model]["metrics"]
            for topic_idx in range(n_topics):
                # automated metrics are at the topic level
                auto_metrics = {
                    k: v[topic_idx] for k, v in metric_data.items()
                    if not isinstance(v[0], list) and
                    k in auto_metric_names
                }

                # build ratings data
                for human_idx in range(n_ratings_annotators):
                    row = {
                        "dataset": dataset,
                        "model": model,
                        "topic_idx": topic_idx,
                        "human_idx": human_idx,
                    }
                    human_metrics = {
                        k.replace("ratings_", ""): v[topic_idx][human_idx] for k, v in metric_data.items()
                        if isinstance(v[0], list) and k.startswith("ratings")
                    }
                    row.update(**auto_metrics)
                    row.update(**human_metrics)
                    ratings_rows.append(row)
                # build intrusion data
                for human_idx in range(n_intrusion_annotators):
                    row = {
                        "dataset": dataset,
                        "model": model,
                        "topic_idx": topic_idx,
                        "human_idx": human_idx,
                    }
                    human_metrics = {
                        k.replace("intrusion_", ""): v[topic_idx][human_idx] for k, v in metric_data.items()
                        if isinstance(v[0], list) and k.startswith("intrusion")
                    }
                    row.update(**auto_metrics)
                    row.update(**human_metrics)
                    intrusion_rows.append(row)

    ratings = pd.DataFrame(ratings_rows)
    intrusions = pd.DataFrame(intrusion_rows)
    ratings["task"] = "ratings"
    intrusions["task"] = "intrusions"

    task_data = pd.concat([ratings, intrusions], ignore_index=True)

    task_data.to_csv(Path(fname.parent, fname.stem + ".csv"), index=False)