import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional, Pattern

import typer
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

from preprocess import read_docs, read_jsonl, docs_to_matrix
from utils import get_total_lines, read_lines

app = typer.Typer()

logger = logging.getLogger(__name__)


class InputFormat(str, Enum):
    text = "text"
    jsonl = "jsonl"


class OutputFormat(str, Enum):
    text = "text"
    binary = "binary"
    sparse = "sparse"


def token_regex_callback(value: str) -> Pattern:
    if value == "alpha":
        return re.compile("[a-zA-Z]")
    if value == "wordlike":
        return re.compile("^[a-zA-Z0-9-_]*[a-zA-Z][a-zA-Z0-9-_]*$")
    if value == "all":
        return None
    return re.compile(value)


@app.command()
def preprocess(
    input_path: list[Path] = typer.Argument(
        ...,
        exists=True,
        help="File(s) containing raw text (works with glob patterns)",
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory. Will save vocabulary file and a document-term matrix."
    ),
    input_format: InputFormat = typer.Option(
        InputFormat.text,
        help="Format of input file(s). If jsonlist, can specify an id variable"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.sparse,
        help="Format in which to save the document-term matrix."
    ),
    lines_are_documents: bool = typer.Option(
        True,
        help="Treat each line in a file as a document (else, each file is a document). "
    ),
    jsonl_text_key: Optional[str] = None,
    jsonl_id_key: Optional[str] = None,

    # Processing
    lowercase: bool = False,
    token_regex: str = typer.Option(
        "alpha",
        callback=token_regex_callback,
        help=(
            "How to retain tokens. \n"
            "`alpha`: keep anything containing at least one letter \n"
            "`wordlike`: keep alphanumeric, eliminating words with all numbers, "
            "or punctuation outside of hyphens, periods, and underscores. \n"
            "`all`: do not filter\n"
            "other values will be interpreted as regex."
        )
    ),
    ngram_range: tuple[int, int] = typer.Option(
        (1, 1),
        help=(
            "Range of ngrams to use. "
            "e.g., (1, 1) is unigrams; (1, 2) unigrams & bigrams; (2, 2) bigrams only. "
            "Not reccomended for use with phrasing options. "
            "(borrowed from sklearn.feature_extraction.text.CountVectorizer)"
        ),
    ),
    remove_stopwords: bool = typer.Option(
        True,
        help="Remove stopwords during processing."
    ),
    min_doc_freq: float = typer.Option(
        1.0,
        min=0,
        help=(
            "Ignore terms with a document frequency lower than this threshold "
            "(if < 1, treated as a proportion of documents)."
            "(borrowed from sklearn.feature_extraction.text.CountVectorizer)"
        ),
    ),
    max_doc_freq: float = typer.Option(
        1.0,
        min=0,
        help=(
            "Ignore terms with a document frequency higher than this threshold, i.e., "
            "corpus-specific stopwords (if < 1, treated as a proportion of documents)."
            "(borrowed from sklearn.feature_extraction.text.CountVectorizer)"
        ),
    ),
    max_vocab_size: Optional[float] = typer.Option(
        None,
        min=0,
        help=( # TODO: change to correspond to sklearn
            "Maximum size of the vocabulary. If < 1, determines how much mass of the"
            "empirical CDF of token counts to retain (in other words, not taking a "
            "share of the unique total tokens by rank)"
        ),
    ),
    detect_entities: bool = typer.Option(
        True,
        help="Automatically detect entities with spaCy, `New York` -> `New_York`."
    ),
    double_count_phrases: bool = typer.Option(
        True,
        help=(
            "Collocations are included alongside constituent unigrams, "
            "`New York -> `New York New_York`. Anecdotally forms more interpretable topics."
        ),
    ),
    
    # External files
    vocabulary: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Use an external vocabulary list. Overrides other preprocessing options."
    ),
    phrases: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="List of known phrases (e.g., `New_York`). Must be connected with underscore"
    ),
    stopwords: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Filepath of stopwords, one word per line. Uses spaCy list if unspecified."
    ),
    encoding: str = "utf8",
):
    # create a doc-by-doc generator
    if input_format.value == "text":
        docs = read_docs(input_path, lines_are_documents, encoding)

    if input_format.value == "jsonl":
        if not lines_are_documents:
            raise ValueError("Input is `jsonl`, but `lines_are_documents` is False")
        if jsonl_text_key is None:
            raise ValueError("Input is `jsonl`, but `jsonl_text_key` unspecified.")
        docs = read_jsonl(input_path, jsonl_text_key, jsonl_id_key, encoding)

    if vocabulary and detect_entities:
        logger.warn(
            "You are detecting entities while also specifying an outside vocabulary "
            "this could mean you filter out discovered entities not in your vocabulary."
        )

    # TODO: make sure lowercase settings & outside word lists are compatible
    # perhaps by checking to see if any uppercase appear?

    # retrieve the total number of documents for progress bars
    total_docs = len(input_path)
    if lines_are_documents:
        total_docs = get_total_lines(input_path, encoding=encoding)

    # load external wordlist files
    vocabulary = set(read_lines(vocabulary, encoding)) if vocabulary else None
    phrases = set(read_lines(phrases, encoding)) if phrases else None
    stopwords = set(read_lines(stopwords, encoding)) if stopwords else STOP_WORDS

    # create the document-term matrix
    dtm = docs_to_matrix(
        docs,
        lowercase=lowercase,
        ngram_range=ngram_range,
        remove_stopwords=remove_stopwords,
        min_doc_freq=min_doc_freq,
        max_doc_freq=max_doc_freq,
        max_vocab_size=max_vocab_size,
        detect_entities=detect_entities,
        double_count_phrases=double_count_phrases,
        token_regex=token_regex,
        vocabulary=vocabulary,
        phrases=phrases,
        stopwords=stopwords,
        total_docs=total_docs,
    )

@app.command()
def run_model():
    pass

if __name__ == "__main__":
    app()