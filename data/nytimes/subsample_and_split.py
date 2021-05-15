import argparse
import random
from pathlib import Path

from tqdm import tqdm

from sklearn.model_selection import train_test_split

def read_lines(path, subsample_pct=1.0, min_doc_size=0, seed=11235,):
    random.seed(seed)
    with open(path, "r") as infile:
        return [
            line.strip() for line in enumerate(infile)
            if len(line.split()) > min_doc_size
            and random.random() < subsample_pct
        ]

def save_text(doc_list, path):
    with open(path, "w") as outfile:
        for i, doc in enumerate(doc_list):
            if i == 0:
                outfile.write(doc)
            else:
                outfile.write("\n" + doc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument("--val-split", type=float)
    parser.add_argument("--test-split", type=float)
    parser.add_argument("--subsample-percentage", type=float, default=1.)
    parser.add_argument("--min_doc_size", type=int, default=0.)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    train_doc_list = read_lines(
        Path(args.data_dir, "articles.txt"),
        min_doc_size=args.min_doc_size,
        subsample_pct=args.subsample_percentage,
    )

    held_out_split = args.val_split + args.test_split

    if args.val_split > 0:
        train_doc_list, test_doc_list = train_test_split(
            train_doc_list, test_size=held_out_split, random_state=11235
        )
    if args.test_split > 0:
        val_doc_list, test_doc_list = train_test_split(
            test_doc_list, test_size=args.test_split / (held_out_split), random_state=11235
        )


    save_text(train_doc_list, Path(args.output_dir, "train.txt"))
    if args.val_split > 0:
        save_text(val_doc_list, Path(args.output_dir, "val.txt"))
    if args.test_split > 0:
        save_text(test_doc_list, Path(args.output_dir, "test.txt"))