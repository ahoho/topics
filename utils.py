import logging
import subprocess
from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)

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


def read_lines(path: Union[Path, str], encoding: str = "utf-8"):
    """
    Read the lines in a file
    """
    with open(path, encoding=encoding) as infile:
        return [line for line in infile if line.strip()]


def gen_ngrams(tokens, min_n, max_n):
    return [
        "_".join(tokens[i:i+n])
        for n in range(min_n, max_n+1)
        for i in range(len(tokens)-n+1)
    ]