import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_argmin_min
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

    wiki_train_dtm = sparse.load_npz(Path(args.input_dir, "train.dtm.npz"))
    wiki_full_dtm = sparse.load_npz(Path(args.input_dir, "full.dtm.npz"))
    wiki_full_ids = load_json(Path(args.input_dir, "full.ids.json"))
    wiki_full_text_file = Path(args.input_dir, "full.txt")

    # use a simple (but computationally-expensive) method to remove 
    # the wikitext training set from the wiki dump
    wikitext_filter_path = Path(args.input_dir, "wikitext_filter.npy")
    if not wikitext_filter_path.exists():
        # warning: very long! use as many processors as possible
        # ~3.5 hours with 45 procs
        print("calculating pairwise distances to make")
        dist_argmin, dist_argmax = pairwise_distances_argmin_min(
            wiki_train_dtm,
            wiki_full_dtm,
            metric='l1',
            metric_kwargs={'working_memory': 1024*16, 'n_jobs': -1},
        )
        np.save(wikitext_filter_path, dist_argmin)
    else:
        dist_argmin = np.load(wikitext_filter_path)
        assert(len(dist_argmin) == wiki_train_dtm.shape[0])
    # make a filter, then filter out training docs from the full set
    wiki_matched_idx = set(dist_argmin)
    wiki_full_filter = [i not in wiki_matched_idx for i in range(wiki_full_dtm.shape[0])]
    
    wiki_full_dtm_filtered = wiki_full_dtm[wiki_full_filter]
    wiki_full_ids_filtered = [id for i, id in enumerate(wiki_full_ids) if wiki_full_filter[i]]

    # make val and test splits
    val_size = int(args.val_split * wiki_train_dtm.shape[0]) if args.val_split <=1 else int(args.val_split)
    wiki_val_dtm, wiki_test_dtm, wiki_val_ids, wiki_test_ids = train_test_split(
        wiki_full_dtm_filtered, wiki_full_ids_filtered, train_size=val_size, random_state=args.seed
    )

    n = len(wiki_full_ids)
    wiki_val_ids_set = set(wiki_val_ids)
    wiki_val_text = (
        doc
        for id, doc in tqdm(zip(wiki_full_ids, load_text(wiki_full_text_file)), total=n)
        if id in wiki_val_ids_set
    )
    wiki_test_ids_set = set(wiki_test_ids)
    wiki_test_text = (
        doc
        for id, doc in tqdm(zip(wiki_full_ids, load_text(wiki_full_text_file)), total=n)
        if id in wiki_test_ids_set
    )

    # save val and test sets
    print("saving splits...")
    sparse.save_npz(Path(args.input_dir, "val.dtm.npz"), wiki_val_dtm)
    save_json(wiki_val_ids, Path(args.input_dir, "val.ids.json"))
    save_text(wiki_val_text, Path(args.input_dir, "val.txt"))

    sparse.save_npz(Path(args.input_dir, "test.dtm.npz"), wiki_test_dtm)
    save_json(wiki_test_ids, Path(args.input_dir, "test.ids.json"))
    save_text(wiki_test_text, Path(args.input_dir, "test.txt"))