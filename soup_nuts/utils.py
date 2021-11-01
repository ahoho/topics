from collections.abc import Iterable
from enum import Enum
import json
import logging
import subprocess
from typing import Union, Any, Optional
from pathlib import Path

from scipy import sparse

logger = logging.getLogger(__name__)


def expand_paths(path_or_pattern):
    """
    Make a list of paths from a glob pattern
    From https://stackoverflow.com/a/51108375
    """
    path = Path(path_or_pattern).expanduser()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    return list(Path(path.root).glob(str(Path("").joinpath(*parts))))


def get_total_lines(paths: list[Union[Path, str]], encoding: str = "utf-8") -> int:
    """
    Get the total number of lines (read: documents) to process
    """
    logger.info("Calculating total number of documents...")
    try:
        # wc is faster than native python
        return sum(
            int(subprocess.check_output(f"/usr/bin/wc -l {p}", shell=True).split()[0])
            for p in paths
        )
    except subprocess.CalledProcessError:
        return sum(1 for p in paths for line in open(p, encoding=encoding))


def read_lines(path: Union[Path, str], encoding: str = "utf-8") -> list[str]:
    """
    Read the lines in a file
    """
    with open(path, encoding=encoding) as infile:
        return [line for line in infile if line.strip()]


def save_lines(obj: Iterable, fpath: Union[str, Path]):
    with open(fpath, "w") as outfile:
        for i, x in enumerate(obj):
            if i == 0:
                outfile.write(x)
            else:
                outfile.write(f"\n{x}")


def save_json(obj: Any, fpath: Union[str, Path], indent: Optional[int] = None):
    with open(fpath, "w") as outfile:
        json.dump(obj, outfile, indent=indent)


def save_params(params: dict[str, Any], fpath: Union[str, Path]):
    safe_params = {}
    safe_types = (float, int, str, bool, type(None))
    for k, v in params.items():
        if isinstance(v, Enum):
            v = v.value
        if not isinstance(v, (tuple, list) + safe_types):
            v = str(v)
        if isinstance(v, (tuple, list)) and any(
            (not isinstance(i, safe_types)) for i in v
        ):
            v = [str(v) for v in v]
        safe_params[k] = v
    save_json(safe_params, fpath, indent=2)


def save_jsonl(
    metadata: list[dict[str, Any]],
    outpath: Union[str, Path],
    dtm: Optional[sparse.csr.csr_matrix] = None,
    vocab: Optional[dict[str, int]] = None,
):
    """
    Save a list of dictionaries to a jsonl file. If `dtm` and `vocab` are provided,
    save document-term matrix as a dictionary in the following format, where each
    row is a document:
    {
        "id": <doc_1>,
        <other metadata>: ...
        "tokenized_counts": {
            <word_2>: <count_of_word_2_in_doc_1>,
            <word_6>: <count_of_word_6_in_doc_1>,
            ...
        },
    }
    """

    if dtm is not None and vocab is None:
        raise ValueError("`vocab` must be provided if `dtm` is provided")

    if vocab is not None : 
        inv_vocab = dict(zip(vocab.values(), vocab.keys()))
    
    with open(outpath, mode="w") as outfile:
        for i, data in enumerate(metadata):
            if dtm is not None and vocab is not None:
                row = dtm[i]
                words_in_doc = [inv_vocab[idx] for idx in row.indices]
                counts = [int(v) for v in row.data]  # int64 not serializable
                word_counts = dict(zip(words_in_doc, counts))
                data.update({"tokenized_counts": word_counts})
            row_json = json.dumps(data)
            if i == 0:
                outfile.write(row_json)
            else:
                outfile.write(f"\n{row_json}")


def gen_ngrams(tokens: list[str], min_n: int, max_n: int) -> list[str]:
    """
    Create all ngrams from `tokens` where n is between `min_n`, `max_n`, inclusive.
    """
    return [
        "_".join(tokens[i : i + n])
        for n in range(min_n, max_n + 1)
        for i in range(len(tokens) - n + 1)
    ]