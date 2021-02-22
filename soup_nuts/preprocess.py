import json
import logging
from re import Pattern
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Union, Optional

from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from tqdm import tqdm

import spacy
from spacy.language import Language
from spacy.tokens import Token

from phrases import make_phrase_matcher, make_phrase_merger
from utils import gen_ngrams

logger = logging.getLogger(__name__)

def read_docs(
    paths: Union[Union[Path, str], list[Union[Path, str]]],
    lines_are_documents: bool = True,
    max_doc_size: Optional[int] = None,
    encoding: str = "utf-8", # TODO: change to system default?
) -> Iterator[tuple[str, str]]:
    """
    Lazily read in contents of files.
    """
    if isinstance(paths, (Path, str)):
        paths = [paths]
    for path in paths:
        with open(path, "r", encoding=encoding) as infile:
            if lines_are_documents:
                for i, text in enumerate(infile):
                    if text:
                        yield _truncate_doc(text, max_doc_size), f"{path}:{i:09}"
            else:
                text = infile.read().strip()
                if text:
                    yield _truncate_doc(text, max_doc_size), f"{path}"


def read_jsonl(
    paths: Union[Union[Path, str], list[Union[Path, str]]],
    text_key: str = "text",
    id_key: Optional[str] = None,
    max_doc_size: Optional[int] = None,
    encoding: str = "utf-8",
) -> Iterator[tuple[str, str]]:
    """
    Lazily read in contents of jsonlist files.
    """
    if isinstance(paths, (Path, str)):
        paths = [paths]
    
    for path in paths:
        with open(path, "r", encoding=encoding) as infile:
            for i, line in enumerate(infile):
                if line:
                    data = json.loads(line)
                    id = str(data[id_key]) if id_key else f"{path}:{i:09}"
                    text = data[text_key].replace("\n", " ") # remove linebreaks
                    yield _truncate_doc(text, max_doc_size), id


def _truncate_doc(
    doc: str,
    max_len: Optional[int] = None,
) -> str:
    """
    Truncate a document to max_len by the rough number of tokens
    """
    if not max_len:
        return doc
    if doc.count(" ") > max_len:
        doc = " ".join(doc.split(" ")[:max_len])
        return doc
    return doc

                    
def docs_to_matrix(
    docs: Iterable[Union[tuple[str, str], str]],
    as_tuples: bool = True,
    lowercase: bool = False,
    ngram_range: tuple[int, int] = (1, 1),
    remove_stopwords: bool = True,
    min_doc_freq: float = 1.0,
    max_doc_freq: float = 1.0,
    max_vocab_size: Optional[float] = None,
    detect_entities: bool = False,
    double_count_phrases: bool = True,
    token_regex: Optional[Pattern] = None,
    vocabulary: Optional[list[str]] = None,
    phrases: Optional[list[str]] = None,
    stopwords: Optional[list[str]] = None,
    total_docs: Optional[int] = None,
    n_process: int = 1,
) -> tuple[sparse.csr.csr_matrix, dict[str, int], list[str]]:
    """
    Create a document-term matrix for a list of documents
    """
    doc_tokens = tokenize_docs(
        docs=docs,
        as_tuples=as_tuples,
        lowercase=lowercase,
        ngram_range=ngram_range,
        remove_stopwords=remove_stopwords,
        detect_entities=detect_entities,
        double_count_phrases=double_count_phrases,
        token_regex=token_regex,
        vocabulary=vocabulary,
        phrases=phrases,
        stopwords=stopwords,
        n_process=n_process,
    )
    if not as_tuples: # add ids if none were used
        doc_tokens = ((doc, i) for i, doc in enumerate(doc_tokens))

    # CountVectorizer is considerably faster than gensim for creating the doc-term mtx
    cv = CountVectorizerWithID(
        preprocessor=lambda x: x,
        analyzer=lambda x: x,
        min_df=float(min_doc_freq) if min_doc_freq < 1 else int(min_doc_freq),
        max_df=float(max_doc_freq) if max_doc_freq <=1 else int(max_doc_freq),
        max_features=max_vocab_size,#TODO, fix how we describe this param
    )
    logger.info("Processing documents...")
    dtm = cv.fit_transform(tqdm(doc_tokens, total=total_docs))
    vocab = dict(sorted(cv.vocabulary_.items(), key=lambda x: x[1]))
    ids = cv.ids
    return dtm, vocab, ids
            

def tokenize_docs(
    docs: Iterable[Union[tuple[str, str], str]],
    as_tuples: bool = True,
    lowercase: bool = False,
    ngram_range: tuple[int, int] = (1, 1),
    remove_stopwords: bool = True,
    detect_entities: bool = False,
    double_count_phrases: bool = False,
    token_regex: Optional[Pattern] = None,
    vocabulary: Optional[list[str]] = None,
    phrases: Optional[list[str]] = None,
    stopwords: Optional[list[str]] = None,
    n_process: int = 1,
) -> Iterator[Union[tuple[str, str], str]]:
    """
    Tokenize a stream of documents. Tries to be performant by using nlp.pipe 
    """
    # initialize the spacy model
    nlp = create_pipeline(
        lowercase=lowercase,
        detect_entities=detect_entities,
        phrases=phrases,
        stopwords=stopwords,
    )

    # converting `spacy.tokens.Token` to a string
    def to_string(x: Token) -> Union[tuple[str, str], tuple[str]]:
        text = x.lower_ if lowercase else x.text
        if double_count_phrases and " " in text:
            return (text.replace(' ', '_'), *text.split())
        else:
            return (text.replace(" ", "_"), )

    # retain only desirable words
    if vocabulary:
        vocabulary = set(vocabulary)
        if phrases:
            vocabulary |= set(phrases)
    
    def to_keep(x: str) -> bool:
        if vocabulary:
            return x in vocabulary
        if remove_stopwords and x in stopwords:
            return False
        if token_regex:
            return token_regex.search(x)

        return True

    # send through the pipe, `as_tuples` carries the ids forward
    for doc in nlp.pipe(docs, as_tuples=as_tuples, n_process=n_process):
        if as_tuples:
            doc, id = doc
        # If using an outside vocabulary, continue apace
        tokens = [text for tok in doc for text in to_string(tok) if to_keep(text)]
        min_n, max_n = ngram_range
        if max_n == 1:
            if as_tuples:
                tokens = tokens, id
            yield tokens
        else:
            ngrams = gen_ngrams(tokens, min_n, max_n)
            if as_tuples:
                ngrams = ngrams, id
            yield ngrams


def create_pipeline(
    model: str = "en_core_web_sm",
    lowercase: bool = False,
    detect_entities: bool = False,
    phrases: Optional[list[str]] = None,
    stopwords: Optional[list[str]] = None,
) -> Language:
    """
    Create the tokenization pipeline. The main changes come with phrase detection.
    """
    nlp = spacy.load(
        model,
        exclude=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'],
        disable='ner'
    )

    # setup phrase detection
    any_phrases = detect_entities or phrases
    if phrases:
        # we want custom phrases detected before NER, in case of overlap
        nlp.add_pipe(
            "match_phrases",
            config={"phrases": phrases, "lowercase": lowercase},
            before="ner",
        )
    if detect_entities:
        nlp.enable_pipe("ner")
    if any_phrases:
        nlp.add_pipe(
            "merge_phrases",
            config={
                "stopwords": stopwords,
                "filter_entities": ['PERSON', 'FACILITY', 'GPE', 'LOC', 'CUSTOM'],
            }
        )

    return nlp


class CountVectorizerWithID(CountVectorizer):
    def fit_transform(
        self,
        raw_documents: Iterable[tuple[str, str]],
    ) -> sparse.csr.csr_matrix:
        """
        Allows us to collect the document ids during processing while still maintaining
        a lazy generator
        """
        docs = self.doc_iterator(raw_documents)
        return super().fit_transform(docs)

    def doc_iterator(
        self,
        raw_documents: Iterable[tuple[str, str]],
    ) -> Iterable[str]:
        self.ids = []
        for doc, id in raw_documents:
            self.ids.append(id)
            yield doc