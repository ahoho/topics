import argparse
import os
import re
from pathlib import Path

from tqdm import tqdm

from sklearn.model_selection import train_test_split

def is_document_start(line):
    if len(line) < 4:
        return False
    if line[0] == '=' and line[-1] == '=':
        if line[2] != '=':
            return True
        else:
            return False
    else:
        return False


def token_list_per_doc(input_dir, token_file):
    """
    Source from github.com/awslabs/w-lda/blob/master/examples/domains/wikitext103_wae.py
    """
    lines_list = []
    line_prev = ''
    prev_line_start_doc = False
    with open(Path(input_dir, token_file), 'r', encoding='utf-8') as f:
        for l in tqdm(f):
            line = l.strip()
            if prev_line_start_doc and line:
                # the previous line should not have been start of a document!
                lines_list.pop()
                lines_list[-1] = lines_list[-1] + ' ' + line_prev

            if line:
                if is_document_start(line) and not line_prev:
                    if line.startswith("="):
                        line = line[1:]
                    lines_list.append(line)
                    prev_line_start_doc = True
                else:
                    lines_list[-1] = lines_list[-1] + ' ' + line
                    prev_line_start_doc = False
            else:
                prev_line_start_doc = False
            line_prev = line
    print("{} documents parsed!".format(len(lines_list)))
    return lines_list


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
    parser.add_argument("--use-raw", action="store_true", default=False)
    parser.add_argument("--rejoin-split-terms", action="store_true", default=False)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    suffix = "tokens"
    if args.use_raw:
        suffix = "raw"

    # download data
    raw_str = "-raw" if suffix == "raw" else ""
    if not Path(f'{args.data_dir}/wiki.train.{suffix}').exists():
        print("downloading wikitext-103 tokens...")
        os.system(f"curl -O https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103{raw_str}-v1.zip")
        os.system(f"unzip wikitext-103{raw_str}-v1.zip")
        os.system(f"mv wikitext-103{raw_str} {args.data_dir}")
        os.system(f"rm wikitext-103{raw_str}-v1.zip")

    train_file = f'wiki.train.{suffix}'
    train_doc_list = token_list_per_doc(args.data_dir, train_file)

    held_out_split = args.val_split + args.test_split

    if args.val_split:
        train_doc_list, test_doc_list = train_test_split(
            train_doc_list, test_size=held_out_split, random_state=11235
        )
    if args.test_split:
        val_doc_list, test_doc_list = train_test_split(
            test_doc_list, test_size=args.test_split / (held_out_split), random_state=11235
        )

    if args.rejoin_split_terms:
        rejoiner = re.compile(r"\s+@([.\-,])@\s+")
        train_doc_list = [rejoiner.sub(r"\1", doc) for doc in train_doc_list]
        if args.val_split:
            val_doc_list = [rejoiner.sub(r"\1", doc) for doc in val_doc_list]
        if args.test_split:
            test_doc_list = [rejoiner.sub(r"\1", doc) for doc in test_doc_list]

    save_text(train_doc_list, Path(args.output_dir, f"train{raw_str}.txt"))
    if args.val_split:
        save_text(val_doc_list, Path(args.output_dir, f"val{raw_str}.txt"))
    if args.test_split:
        save_text(test_doc_list, Path(args.output_dir, f"test{raw_str}.txt"))