from collections.abc import Iterable, Iterator
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional, Iterable, Union

import typer
from scipy import sparse
from spacy.lang.en.stop_words import STOP_WORDS

from .preprocess import read_docs, read_jsonl, docs_to_matrix
from .phrases import detect_phrases as detect_phrases_
from .utils import (
    get_total_lines,
    read_lines,
    save_lines,
    save_json,
    save_dtm_as_jsonl,
    save_params,
)

app = typer.Typer()

logger = logging.getLogger(__name__)


class InputFormat(str, Enum):
    text: str = "text"
    jsonl: str = "jsonl"


class OutputFormat(str, Enum):
    sparse: str = "sparse"
    jsonl: str = "jsonl"


def token_regex_callback(value: str) -> re.Pattern:
    if value == "alpha":
        return re.compile("[a-zA-Z]")
    if value == "wordlike":
        return re.compile("^[a-zA-Z0-9-_]*[a-zA-Z][a-zA-Z0-9-_]*$")
    if value == "all":
        return None
    return re.compile(value)


def stopwords_callback(value: str) -> Iterable[str]:
    if value == "english":
        return list(STOP_WORDS)
    if value == "none":
        return None
    return read_lines(value)


@app.command(help="Preprocess documents to a document-term matrix.")
def preprocess(
    input_path: list[Path] = typer.Argument(
        ...,
        exists=True,
        help="File(s) containing raw text (works with glob patterns)",
    ),
    output_dir: Path = typer.Argument(
        ..., help="Output directory. Will save vocabulary and the document-term matrix."
    ),
    input_format: InputFormat = typer.Option(
        InputFormat.text,
        help="Format of input file(s). If jsonlist, can specify an id variable",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.sparse,
        help=(
            "Format in which to save the document-term matrix. "
            "For `jsonl`, each row contains the document's id and word counts, "
            "e.g. {'id': '0001', 'counts': {'apple': 2, 'cake': 1}. "
            "`sparse` is a `scipy.sparse.csr.csr_matrix`, vocab & ids in separate files"
        ),
    ),
    lines_are_documents: bool = typer.Option(
        True,
        help="Treat each line in a file as a document (else, each file is a document). ",
    ),
    jsonl_text_key: Optional[str] = None,
    jsonl_id_key: Optional[str] = None,
    # Processing
    lowercase: bool = False,
    token_regex: str = typer.Option(
        "alpha",
        callback=token_regex_callback,
        help=(
            "How to retain tokens: "
            "`alpha`: keep anything containing at least one letter. "
            "`wordlike`: keep alphanumeric, eliminating words with all numbers, "
            "or punctuation outside of hyphens, periods, and underscores. "
            "`all`: do not filter "
            "other values will be interpreted as regex."
        ),
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
    min_doc_freq: float = typer.Option(
        1,
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
            "corpus-specific stopwords (if <= 1, treated as a proportion of documents)."
            "(borrowed from sklearn.feature_extraction.text.CountVectorizer)"
        ),
    ),
    max_vocab_size: Optional[int] = typer.Option(
        None,
        min=0,
        help="Maximum size of the vocabulary. If < 1, share of total vocab to keep",
    ),
    detect_entities: bool = typer.Option(
        True,
        help=("Automatically detect entities with spaCy, `New York` -> `New_York`. "),
    ),
    double_count_phrases: bool = typer.Option(
        True,
        help=(
            "Collocations are included alongside constituent unigrams, "
            "`New York -> `New York New_York`. Anecdotally forms more interpretable topics."
        ),
    ),
    max_doc_size: Optional[int] = typer.Option(
        None, min=0, help="Maximum document size in whitespace-delimited tokens"
    ),
    # External files
    vocabulary: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Use an external vocabulary list. Overrides other preprocessing options.",
    ),
    phrases: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Filepath of known phrases (e.g., `New_York`). Must be connected with underscore",
    ),
    stopwords: Optional[str] = typer.Option(
        "english",
        help=(
            "Filepath of stopwords, one word per line. "
            "Set to `english` to use spaCy defaults or `none` to not remove stopwords",
        )
    ),
    encoding: str = "utf8",
    n_process: int = -1,
):
    params = locals()

    # create a doc-by-doc generator
    if input_format.value == "text":
        docs = read_docs(input_path, lines_are_documents, max_doc_size, encoding)

    if input_format.value == "jsonl":
        if not lines_are_documents:
            raise ValueError("Input is `jsonl`, but `lines_are_documents` is False")
        if jsonl_text_key is None:
            raise ValueError("Input is `jsonl`, but `jsonl_text_key` unspecified.")
        docs = read_jsonl(
            input_path, jsonl_text_key, jsonl_id_key, max_doc_size, encoding
        )

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
    vocabulary = read_lines(vocabulary, encoding) if vocabulary else None
    phrases = read_lines(phrases, encoding) if phrases else None

    # create the document-term matrix
    dtm, vocab, ids = docs_to_matrix(
        docs,
        lowercase=lowercase,
        ngram_range=ngram_range,
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
        n_process=n_process,
    )

    # save out
    save_params(params, Path(output_dir, "params.json"))
    if output_format == "sparse":
        sparse.save_npz(Path(output_dir, "dtm.npz"), dtm)
        save_json(vocab, Path(output_dir, "vocab.json"), indent=2)
        save_json(ids, Path(output_dir, "ids.json"), indent=2)
    if output_format == "jsonl":
        save_dtm_as_jsonl(dtm, vocab, ids, Path(output_dir, "data.jsonl"))


def connector_words_callback(value: str) -> Iterable[str]:
    from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS

    if value == "english":
        return ENGLISH_CONNECTOR_WORDS | STOP_WORDS
    if value == "gensim_default":
        return ENGLISH_CONNECTOR_WORDS
    if value == "none":
        return None
    return read_lines(value)


@app.command(help="Learn phrases (e.g., `New_York`) from the data.")
def detect_phrases(
    input_path: list[Path] = typer.Argument(
        ...,
        exists=True,
        help="File(s) containing raw text (works with glob patterns)",
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory. Will save vocabulary file and a document-term matrix.",
    ),
    input_format: InputFormat = typer.Option(
        InputFormat.text,
        help="Format of input file(s). If jsonlist, can specify an id variable",
    ),
    lines_are_documents: bool = typer.Option(
        True,
        help="Treat each line in a file as a document (else, each file is a document). ",
    ),
    jsonl_text_key: Optional[str] = None,
    jsonl_id_key: Optional[str] = None,
    max_doc_size: Optional[int] = typer.Option(
        None, min=0, help="Maximum document size in whitespace-delimited tokens"
    ),
    passes: int = typer.Option(
        1,
        help=(
            "Passes over the data, where more passes leads to longer phrase detection, "
            "e.g., pass one yields `New_York`, pass two yields `New_York_City`."
        ),
    ),
    lowercase: bool = False,
    detect_entities: bool = typer.Option(
        True,
        help=(
            "Automatically detect entities with spaCy, `New York` -> `New_York`. "
            "If you plan to set this to `True` during preprocessing, do it now "
            "and set it to `False` after"
        ),
    ),
    token_regex: str = typer.Option(
        "alpha",
        callback=token_regex_callback,
        help=(
            "How to retain tokens: "
            "`alpha`: keep anything containing at least one letter. "
            "`wordlike`: keep alphanumeric, eliminating words with all numbers, "
            "or punctuation outside of hyphens, periods, and underscores. "
            "`all`: do not filter "
            "other values will be interpreted as regex."
        ),
    ),
    min_count: int = 5,
    threshold: float = 10.0,
    max_vocab_size: float = 40_000_000,
    connector_words: Optional[str] = typer.Option(
        "english",
        callback=connector_words_callback,
        help=(
            "Point to a path of connector words, or use `english` to use common english "
            "articles (from gensim). Set to `none` to not use. From gensim docs: "
            "'Set of words that may be included within a phrase, without affecting its "
            "scoring. No phrase can start nor end with a connector word; a phrase may "
            "contain any number of connector words in the middle.' "
        ),
    ),
    phrases: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "Path to list of already-known phrases (e.g., `New_York`). "
            "Must be connected with underscore."
        ),
    ),
    encoding="utf-8",
    n_process: int = -1,
):
    params = locals()

    # create a doc-by-doc generator
    if input_format.value == "text":
        read_docs_ = lambda: read_docs(
            input_path, lines_are_documents, max_doc_size, encoding
        )

    if input_format.value == "jsonl":
        if not lines_are_documents:
            raise ValueError("Input is `jsonl`, but `lines_are_documents` is False")
        if jsonl_text_key is None:
            raise ValueError("Input is `jsonl`, but `jsonl_text_key` unspecified.")
        read_docs_ = lambda: read_jsonl(
            input_path, jsonl_text_key, jsonl_id_key, max_doc_size, encoding
        )

    # retrieve the total number of documents for progress bars
    total_docs = len(input_path)
    if lines_are_documents:
        total_docs = get_total_lines(input_path, encoding=encoding)

    phrases = read_lines(phrases, encoding) if phrases else None

    phrases = detect_phrases_(
        docs_reader=read_docs_,
        passes=passes,
        lowercase=lowercase,
        detect_entities=detect_entities,
        token_regex=token_regex,
        min_count=min_count,
        threshold=threshold,
        max_vocab_size=max_vocab_size,
        connector_words=connector_words,
        phrases=phrases,
        n_process=n_process,
        total_docs=total_docs,
    )
    save_params(params, Path(output_dir, "params.json"))
    save_lines(phrases.keys(), Path(output_dir, "phrases.json"))


@app.command()
def run_model():
    pass