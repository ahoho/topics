import json
import logging
from re import Pattern
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Union, Optional
from numpy import isin

from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from tqdm import tqdm

import spacy
from spacy.language import Language
from spacy.tokens import Token

from .phrases import make_phrase_matcher, make_phrase_merger
from .utils import gen_ngrams

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
    lowercase: bool = False,
    ngram_range: tuple[int, int] = (1, 1),
    min_doc_freq: float = 1.0,
    max_doc_freq: float = 1.0,
    max_vocab_size: Optional[float] = None,
    detect_entities: bool = False,
    detect_noun_chunks: bool = False,
    double_count_phrases: bool = True,
    max_phrase_len: Optional[int] = None,
    token_regex: Optional[Pattern] = None,
    min_chars: Optional[int] = None,
    vocabulary: Optional[Iterable[str]] = None,
    phrases: Optional[Iterable[str]] = None,
    stopwords: Optional[Iterable[str]] = None,
    total_docs: Optional[int] = None,
    spacy_model: Union[Language, str] = 'en_core_web_sm',
    as_tuples: bool = True,
    n_process: int = 1,
) -> tuple[sparse.csr.csr_matrix, dict[str, int], list[str]]:
    """
    Create a document-term matrix for a list of documents
    """
    doc_tokens = tokenize_docs(
        docs=docs,
        lowercase=lowercase,
        ngram_range=ngram_range,
        detect_entities=detect_entities,
        detect_noun_chunks=detect_noun_chunks,
        double_count_phrases=double_count_phrases,
        max_phrase_len=max_phrase_len,
        token_regex=token_regex,
        min_chars=min_chars,
        vocabulary=vocabulary,
        phrases=phrases,
        stopwords=stopwords,
        spacy_model=spacy_model,
        as_tuples=as_tuples,
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
    lowercase: bool = False,
    ngram_range: tuple[int, int] = (1, 1),
    detect_entities: bool = False,
    detect_noun_chunks: bool = False,
    double_count_phrases: bool = False,
    max_phrase_len: Optional[int] = None,
    token_regex: Optional[Pattern] = None,
    min_chars: Optional[int] = None,
    vocabulary: Optional[Iterable[str]] = None,
    phrases: Optional[Iterable[str]] = None,
    stopwords: Optional[Iterable[str]] = None,
    spacy_model: Union[Language, str] = 'en_core_web_sm',
    as_tuples: bool = True,
    n_process: int = 1,
) -> Iterator[Union[tuple[str, str], str]]:
    """
    Tokenize a stream of documents. Tries to be performant by using nlp.pipe 
    """
    # initialize the spacy model if it's not already
    if isinstance(spacy_model, str):
        spacy_model = create_pipeline(
            model_name=spacy_model,
            detect_entities=detect_entities,
            detect_noun_chunks=detect_noun_chunks,
            case_sensitive=not lowercase,
            phrases=phrases,
            phrase_stopwords=stopwords,
            max_phrase_len=max_phrase_len,
        )
    else:
        any_phrases = detect_entities or detect_noun_chunks or phrases
        if any_phrases and "merge_phrases" not in spacy_model.pipe_names:
            logger.warn(
                "Your `spacy_model` is missing the `merge_phrases` pipe but `phrases`, "
                "`detect_entities`, or `detect_noun_chunks` is in use. Phrases will not "
                "be included in tokenization. "
            )

    # augment vocabulary with any given phrases
    if vocabulary:
        vocabulary = set(vocabulary)
        if phrases:
            vocabulary |= set(phrases)
        if lowercase:
            vocabulary = {v.lower() for v in vocabulary}

    # how to convert `spacy.tokens.Token` to tuples of strings
    def to_string(x: Token) -> Union[tuple[str, str], tuple[str]]:
        text = x.lower_ if lowercase else x.text
        if double_count_phrases and " " in text:
            return (text.replace(" ", "_"), *text.split(" "))
        else:
            return (text.replace(" ", "_"), )
    
    # determine which tokens will be retained
    def to_keep(x: str) -> bool:
        if vocabulary:
            return x in vocabulary
        if min_chars and len(x) <= min_chars:
            return False
        if stopwords and x in stopwords:
            return False
        if token_regex:
            return token_regex.search(x)
        return True

    # send through the pipe, `as_tuples` carries the ids forward
    for doc in spacy_model.pipe(docs, as_tuples=as_tuples, n_process=n_process):
        if as_tuples:
            doc, id = doc
        
        tokens = [text for tok in doc for text in to_string(tok) if to_keep(text)]
        if not tokens:
            continue

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
    model_name: str = "en_core_web_sm",
    detect_entities: bool = False,
    detect_noun_chunks: bool = False,
    case_sensitive: bool = True,
    phrases: Optional[Iterable[str]] = None,
    phrase_stopwords: Optional[Iterable[str]] = None,
    max_phrase_len: Optional[int] = None,
    filter_entities: Optional[list[str]] = ['ORG', 'PERSON', 'FACILITY', 'GPE', 'LOC'],
) -> Language:
    """
    Create the tokenization pipeline. The main changes come with phrase detection.

    `case_sensitive` determines whether `phrases` are matched exactly

    `phrase_stopwords` specifies words (stopword-like) that can exist inside a phrase
    but not on either side of it: `statue_of_liberty` but not `the_statue_of_liberty`
    """
    nlp = spacy.load(
        model_name,
        exclude=['lemmatizer'],
        disable=['ner', 'tagger', 'parser', 'tok2vec', 'attribute_ruler'],
    )

    # setup phrase detection
    if phrases:
        # we want custom phrases detected before NER, in case of overlap
        nlp.add_pipe(
            "match_phrases",
            config={"phrases": phrases, "case_sensitive": case_sensitive},
            before="ner",
        )
        # if it is None, then all entities are kept anyway in `merge_phrases`
        if filter_entities is not None:
            filter_entities.append("CUSTOM")

    if detect_entities:
        nlp.enable_pipe("ner")
    if detect_noun_chunks:
        nlp.enable_pipe("tok2vec")
        nlp.enable_pipe("attribute_ruler")
        nlp.enable_pipe("tagger")
        nlp.enable_pipe("parser")

    any_phrases = detect_entities or detect_noun_chunks or phrases
    if any_phrases:
        nlp.add_pipe(
            "merge_phrases",
            config={
                "stopwords": list(phrase_stopwords) if phrase_stopwords else None,
                "filter_entities": filter_entities,
                "max_phrase_len": max_phrase_len, # will not impact custom phrases
            },
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