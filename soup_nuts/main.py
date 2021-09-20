from collections.abc import Iterable
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional, Iterable

import typer
from scipy import sparse
from spacy.lang.en.stop_words import STOP_WORDS

from .preprocess import read_docs, read_jsonl, docs_to_matrix
from .phrases import detect_phrases as detect_phrases_
from .utils import (
    expand_paths,
    get_total_lines,
    read_lines,
    save_lines,
    save_json,
    save_jsonl,
    save_params,
)

app = typer.Typer()

logger = logging.getLogger(__name__)


class InputFormat(str, Enum):
    text: str = "text"
    jsonl: str = "jsonl"


def token_regex_callback(value: str) -> re.Pattern:
    if value == "alpha":
        return re.compile("[a-zA-Z]")
    if value == "alphanum":
        return re.compile("\w", flags=re.UNICODE)
    if value == "wordlike":
        return re.compile("^[\w-]*[a-zA-Z][\w-]*$", flags=re.UNICODE)
    if value == "all":
        return None
    return re.compile(value, flags=re.UNICODE)


def stopwords_callback(value: str) -> Iterable[str]:
    if value == "english":
        return STOP_WORDS
    if value == "none":
        return None
    return read_lines(value)


@app.command(help="Preprocess documents to a document-term matrix.")
def preprocess(
    input_path: Optional[Path] = typer.Argument(
        ...,
        exists=True,
        help="File(s) containing text data (works with glob patterns if quoted)",
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory. Will save vocabulary and the document-term matrix.",
    ),
    val_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        help=(
            "Optional file(s) containing raw text to use as validation data. "
            "Will rely on vocabulary from the training data."
        ),
    ),
    test_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        help=(
            "Optional file(s) containing raw text to use as test data. "
            "Will rely on vocabulary from the training data."
        ),
    ),
    input_format: InputFormat = typer.Option(
        InputFormat.text,
        help="Format of input file(s). If jsonlist, can specify an id variable",
    ),

    output_text: bool = typer.Option(False, help="Also output processed text, in order"),
    lines_are_documents: bool = typer.Option(
        True,
        help="Treat each line in a file as a document (else, each file is a document). ",
    ),
    jsonl_text_key: Optional[str] = None,
    jsonl_id_key: Optional[str] = typer.Option(
        None,
        help=(
            "Unique document id for each row in a jsonl. "
            "Will be generated automatically if not specified.",
        ),
    ),
    jsonl_metadata_keys: Optional[str] = typer.Option(
        None,
        help=(
            "Other keys to retain, which will be output in a jsonl file. "
            "Separate with commas, e.g., `key1,key2,key3` "
            "If using models with knowledge distillation, include raw text"
        ),
    ),

    # Processing
    lowercase: bool = False,
    token_regex: str = typer.Option(
        "alphanum",
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
    min_chars: int = typer.Option(1, help="Minimum number of characters per word."),
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
        help="Maximum size of the vocabulary."
    ),
    detect_entities: bool = typer.Option(
        False,
        help="Automatically detect entities with spaCy, `New York` -> `New_York`. "
    ),
    detect_noun_chunks: bool = typer.Option(
        False,
        help=(
            "Automatically detect noun chunks with spaCy "
            "`8.1 million American adults` -> `8.1_million_American_adults` "
            "(will slow down processing)"
        ),
    ),
    double_count_phrases: bool = typer.Option(
        True,
        help=(
            "Collocations are included alongside constituent unigrams, "
            "`New York -> `New York New_York`. Anecdotally forms more interpretable topics."
        ),
    ),
    max_phrase_len: Optional[int] = typer.Option(
        None,
        min=2,
        help=(
            "Maximum length of *detected* phrases in words (not applied to those in `--phrases`) "
            "Currently, this is 'all or nothing': a long phrase will be merely processed "
            "as its component tokens, not reduced. Use `detect-phrases` command for "
            "more fine-grained control"
        ),
    ),
    max_doc_size: Optional[int] = typer.Option(
        None, min=0, help="Maximum document size in whitespace-delimited tokens"
    ),
    min_doc_size: Optional[int] = typer.Option(
        1, min=0, help="Minimum document size",
    ),
    lemmatize: bool = typer.Option(
        False,
        help=(
            "Lemmatize words using the spaCy model, `I am happy` -> `I be happy`. "
            "Will _not_ lemmatize terms inside a phrase, so "
            "`I support voting rights` -> `I support voting_rights`. "
            "It also supersedes vocabulary filtering, so having `words` in `vocabulary` "
            "will mean that the lemmatized `word` is not included (this behavior subject to change). "
            "Finally, note that spaCy is not infallible! "
            "e.g., it will lemmatize `taxes`->`taxis` instead of `tax`."
        ),
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
        callback=stopwords_callback,
        help=(
            "Filepath of stopwords, one word per line. Set to `english` to use spaCy "
            "list (the default) or `none` to not remove stopwords"
        )
    ),  
    # TODO: whitespace tokenization option
    passthrough: Optional[bool] = typer.Option(
        False,
        help=(
            "Ignore all settings and whitespace tokenize "
            "(vocabulary frequency filters can still be applied: "
            "`vocabulary`, `min_doc_freq`, `max_doc_freq`, `max_vocab_size`) "
        ),
    ),
    encoding: str = "utf8",
    n_process: int = -1,
):
    params = locals()

    input_path = expand_paths(input_path)
    val_path = expand_paths(val_path) if val_path else []
    test_path = expand_paths(test_path) if test_path else []

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if set(input_path) & set(val_path) or set(input_path) & set(test_path) or set(val_path) & set(test_path):
        raise ValueError("There is overlap between the train, test, and validation paths")

    # create a doc-by-doc generator
    if input_format.value == "text":
        docs = read_docs(input_path, lines_are_documents, max_doc_size, encoding)
        val_docs = read_docs(val_path, lines_are_documents, max_doc_size, encoding)
        test_docs = read_docs(test_path, lines_are_documents, max_doc_size, encoding)
        
    if input_format.value == "jsonl":
        if not lines_are_documents:
            raise ValueError("Input is `jsonl`, but `lines_are_documents` is False")
        if jsonl_text_key is None:
            raise ValueError("Input is `jsonl`, but `jsonl_text_key` unspecified.")
        jsonl_metadata_keys = jsonl_metadata_keys.split(",") if jsonl_metadata_keys else None
        docs = read_jsonl(input_path, jsonl_text_key, jsonl_id_key, jsonl_metadata_keys, max_doc_size, encoding)
        val_docs = read_jsonl(val_path, jsonl_text_key, jsonl_id_key, jsonl_metadata_keys, max_doc_size, encoding)
        test_docs = read_jsonl(test_path, jsonl_text_key, jsonl_id_key, jsonl_metadata_keys, max_doc_size, encoding)

    if vocabulary and (detect_entities or detect_noun_chunks):
        logger.warn(
            "You are detecting entities while also specifying an outside vocabulary "
            "this could mean you filter out discovered entities not in your vocabulary."
        )

    if (
        token_regex.search("word_word") is None and 
        (detect_entities or detect_noun_chunks or phrases)
    ):
        raise ValueError(
            "Your `token_regex` does not accept tokens with underscores (`_`), but "
            "`detect_entities`, `detect_noun_chunks` or `phrases` is in use. "
            "Try updating your regex to accommodate underscores."
        )
    # TODO: make sure lowercase settings & outside word lists are compatible
    # perhaps by checking to see if any uppercase appear?

    # retrieve the total number of documents for progress bars
    total_docs, total_val, total_test = len(input_path), len(val_path), len(test_path)
    if lines_are_documents:
        total_docs = get_total_lines(input_path, encoding=encoding)
        total_val = get_total_lines(val_path, encoding=encoding)
        total_test = get_total_lines(test_path, encoding=encoding)

    # load external wordlist files
    vocabulary = read_lines(vocabulary, encoding) if vocabulary else None
    phrases = read_lines(phrases, encoding) if phrases else None

    # create the document-term matrix
    logger.info("Processing train data")
    dtm, terms, metadata = docs_to_matrix(
        docs,
        lowercase=lowercase,
        ngram_range=ngram_range,
        min_doc_freq=min_doc_freq,
        max_doc_freq=max_doc_freq,
        max_vocab_size=max_vocab_size,
        detect_entities=detect_entities,
        detect_noun_chunks=detect_noun_chunks,
        double_count_phrases=double_count_phrases,
        max_phrase_len=max_phrase_len,
        token_regex=token_regex,
        min_chars=min_chars,
        min_doc_size=min_doc_size,
        lemmatize=lemmatize,
        vocabulary=vocabulary,
        phrases=phrases,
        stopwords=stopwords,
        passthrough=passthrough,
        total_docs=total_docs,
        retain_text=output_text,
        n_process=n_process,
    )

    if val_path or test_path:
        # on second pass, keep only _learned_ phrases
        learned_phrases = [v for v in terms if ("_" in v or "-" in v) and v not in (vocabulary or [])]

    # TODO: create data with a single function, then loop through the train/val/test splits
    if val_path:
        logger.info("Processing validation data")
        val_dtm, _, val_metadata = docs_to_matrix(
            val_docs,
            lowercase=lowercase,
            ngram_range=ngram_range,
            double_count_phrases=double_count_phrases, # TODO: may need to make false?
            token_regex=token_regex,
            min_chars=min_chars,
            min_doc_size=min_doc_size,
            lemmatize=lemmatize,
            vocabulary=terms,
            phrases=learned_phrases,
            passthrough=passthrough,
            total_docs=total_val,
            retain_text=output_text,
            n_process=n_process,
        )
    if test_path:
        logger.info("Processing test data")
        test_dtm, _, test_metadata = docs_to_matrix(
            test_docs,
            lowercase=lowercase,
            ngram_range=ngram_range,
            double_count_phrases=double_count_phrases, # TODO: may need to make false?
            token_regex=token_regex,
            min_chars=min_chars,
            min_doc_size=min_doc_size,
            lemmatize=lemmatize,
            vocabulary=terms,
            phrases=learned_phrases,
            passhrough=passthrough,
            total_docs=total_test,
            retain_text=output_text,
            n_process=n_process,
        )
    # save outputs
    save_params(params, Path(output_dir, "params.json"))
    save_json(terms, Path(output_dir, "vocab.json"), indent=2)

    # creates, for each split,
    # - a sparse matrix
    # - a jsonl containing metadata (at the very lease, ids), one row per document
    sparse.save_npz(Path(output_dir, "train.dtm.npz"), dtm)
    save_jsonl(metadata, Path(output_dir, "train.metadata.jsonl"))
    if val_path:
        sparse.save_npz(Path(output_dir, "val.dtm.npz"), val_dtm)
        save_jsonl(val_metadata, Path(output_dir, "val.metadata.jsonl"))
    if test_path:
        sparse.save_npz(Path(output_dir, "test.dtm.npz"), test_dtm)
        save_jsonl(test_metadata, Path(output_dir, "test.metadata.jsonl"))


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
    input_path: Optional[Path] = typer.Argument(
        ...,
        exists=True,
        help="File(s) containing text data (works with glob patterns if quoted)",
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory. Will save vocabulary and the document-term matrix.",
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
    detect_noun_chunks: bool = typer.Option(
        False,
        help=(
            "Automatically detect noun chunks with spaCy, `pinky promise` -> `pinky_promise`. "
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
            "articles (from spaCy), or `gensim_default` for a smaller list from gensim. "
            " Set to `none` to not use. From gensim docs: "
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
    max_phrase_len: Optional[int] = typer.Option(None, help="Max length in tokens"),
    encoding="utf-8",
    n_process: int = -1,
):
    params = locals()

    input_path = expand_paths(input_path)

    # create a doc-by-doc generator
    if input_format.value == "text":
        docs_reader = lambda: read_docs(
            input_path, lines_are_documents, max_doc_size, encoding
        )

    if input_format.value == "jsonl":
        if not lines_are_documents:
            raise ValueError("Input is `jsonl`, but `lines_are_documents` is False")
        if jsonl_text_key is None:
            raise ValueError("Input is `jsonl`, but `jsonl_text_key` unspecified.")
        docs_reader = lambda: read_jsonl(
            input_path, jsonl_text_key, jsonl_id_key, max_doc_size, encoding
        )

    # retrieve the total number of documents for progress bars
    total_docs = len(input_path)
    if lines_are_documents:
        total_docs = get_total_lines(input_path, encoding=encoding)

    phrases = read_lines(phrases, encoding) if phrases else None

    phrases = detect_phrases_(
        docs_reader=docs_reader,
        passes=passes,
        lowercase=lowercase,
        detect_entities=detect_entities,
        detect_noun_chunks=detect_noun_chunks,
        token_regex=token_regex,
        min_count=min_count,
        threshold=threshold,
        max_vocab_size=max_vocab_size,
        connector_words=connector_words,
        phrases=phrases,
        max_phrase_len=max_phrase_len,
        n_process=n_process,
        total_docs=total_docs,
    )
    save_params(params, Path(output_dir, "params.json"))
    save_lines(phrases, Path(output_dir, "phrases.json"))


if __name__ == "__main__":
    app()