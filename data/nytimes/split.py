import argparse
import json
from pathlib import Path

from scipy import sparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def load_text(inpath):
    with open(inpath, "r") as infile:
        for line in infile:
            yield line


def save_text(lines, outpath):
    with open(outpath, "w") as outfile:
        for line_no, line in enumerate(lines):
            if line_no == 0:
                outfile.write(line.strip())
            else:
                outfile.write("\n"+line.strip())


def load_json(inpath):
    with open(inpath, "r") as infile:
        return json.load(infile)


def save_json(obj, outpath):
    with open(outpath, "w") as outfile:
        return json.dump(obj, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--val_split", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # load in train ids and full data
    nyt_train_ids = load_json(Path(args.input_dir, "train.ids.json"))

    nyt_full_dtm = sparse.load_npz(Path(args.input_dir, "full.dtm.npz"))
    nyt_full_ids = load_json(Path(args.input_dir, "full.ids.json"))
    nyt_full_text_file = Path(args.input_dir, "full.txt")

    # make a filter, then filter out the training docs from the full set
    assert(len(set(nyt_train_ids)) == len(nyt_train_ids))
    nyt_train_ids = set(nyt_train_ids)

    nyt_full_dtm_filtered = nyt_full_dtm[[id not in nyt_train_ids for id in nyt_full_ids]]
    nyt_full_ids_filtered = [id for id in nyt_full_ids if id not in nyt_train_ids]

    # make val and test sets
    val_size = int(args.val_split * len(nyt_train_ids)) if args.val_split <=1 else int(args.val_split)
    nyt_val_dtm, nyt_test_dtm, nyt_val_ids, nyt_test_ids = train_test_split(
        nyt_full_dtm_filtered, nyt_full_ids_filtered, train_size=val_size, random_state=args.seed
    )

    n = len(nyt_full_ids)
    nyt_val_ids_set = set(nyt_val_ids)
    nyt_val_text = (
        doc
        for id, doc in tqdm(zip(nyt_full_ids, load_text(nyt_full_text_file)), total=n)
        if id in nyt_val_ids_set
    )
    nyt_test_ids_set = set(nyt_test_ids)
    nyt_test_text = (
        doc
        for id, doc in tqdm(zip(nyt_full_ids, load_text(nyt_full_text_file)), total=n)
        if id in nyt_test_ids_set
    )

    # save val and test sets
    print("saving splits...")
    sparse.save_npz(Path(args.input_dir, "val.dtm.npz"), nyt_val_dtm)
    save_json(nyt_val_ids, Path(args.input_dir, "val.ids.json"))
    save_text(nyt_val_text, Path(args.input_dir, "val.txt"))

    sparse.save_npz(Path(args.input_dir, "test.dtm.npz"), nyt_test_dtm)
    save_json(nyt_test_ids, Path(args.input_dir, "test.ids.json"))
    save_text(nyt_test_text, Path(args.input_dir, "test.txt"))