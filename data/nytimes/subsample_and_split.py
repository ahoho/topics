import argparse
import random
import json
from pathlib import Path

from tqdm import tqdm

def read_lines(path, subsample_pct=1.0, min_doc_size=0, seed=11235,):
    random.seed(seed)
    with open(path, "r") as infile:
        for id, line in enumerate(tqdm(infile)):
            if len(line.split()) > min_doc_size and random.random() < subsample_pct:
                yield id, line

def save_jsonl(doc_list, path):
    with open(path, "w") as outfile:
        for line_no, (id, doc) in enumerate(doc_list):
            json_doc = json.dumps({"id": id, "text": doc})
            if line_no == 0:
                outfile.write(json_doc)
            else:
                outfile.write("\n" + json_doc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument("--subsample-percentage", type=float, default=1.)
    parser.add_argument("--min_doc_size", type=int, default=0.)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    train_doc_list = read_lines(
        Path(args.data_dir, "articles.txt"),
        min_doc_size=args.min_doc_size,
        subsample_pct=args.subsample_percentage,
    )
    full_doc_list = read_lines(
        Path(args.data_dir, "articles.txt"),
        min_doc_size=args.min_doc_size,
        subsample_pct=1.,
    )

    save_jsonl(train_doc_list, Path(args.output_dir, "train.jsonl"))
    save_jsonl(full_doc_list, Path(args.output_dir, "full.jsonl"))