from collections.abc import Iterable
from re import Pattern
import logging
from typing import Callable, Optional, Union

from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans
from spacy.lang.en.stop_words import STOP_WORDS

from tqdm import tqdm

logger = logging.getLogger(__name__)


@Language.factory(
    "match_phrases",
    default_config={
        "lowercase": False,
    },
)
def make_phrase_matcher(
    nlp: Language,
    name: str,
    *,
    phrases: Iterable[str],
    lowercase: bool = False,
):
    return PipedPhraseMatcher(nlp, phrases, lowercase)


class PipedPhraseMatcher:
    def __init__(
        self,
        nlp: Language,
        phrases: Iterable[str],
        lowercase: bool = False,
    ):
        """
        Basically a "custom" NER pipeline relying on a predefined phraselist.

        If these are to take priority (in case of span overlap), run before standard NER.

        `lowercase` does not transform the labels, but determines whether we lowercase
        when checking for membership in `phrases`

        Borrowed from spacy.io/usage/rule-based-matching#phrasematcher
        """
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER" if lowercase else "ORTH")
        patterns = [nlp.make_doc(c.replace("_", " ")) for c in phrases]
        self.matcher.add("phrase_list", patterns)

    def __call__(self, doc: Doc) -> Doc:
        ents = [
            Span(doc, start=s, end=e, label="CUSTOM") for _, s, e in self.matcher(doc)
        ]
        ents = filter_spans(ents)
        doc.set_ents(ents)
        return doc


@Language.factory(
    "merge_phrases",
    default_config={
        "stopwords": STOP_WORDS,
        "filter_entities": {"PERSON", "FACILITY", "GPE", "LOC", "CUSTOM"},
    },
    retokenizes=True,
)
def make_phrase_merger(
    nlp: Language,
    name: str,
    *,
    stopwords: Optional[list[str]] = None,
    filter_entities: Optional[list[str]] = None,
):
    return PhraseMerger(stopwords, filter_entities)


class PhraseMerger:
    def __init__(
        self,
        stopwords: Optional[list[str]] = None,
        filter_entities: Optional[list[str]] = None,
    ):
        """
        Stopwords, and entity labels are preferred as sets since lookup is O(1).
        """
        self.stopwords = set(stopwords) if stopwords else None
        self.filter_entities = set(filter_entities)

    def __call__(self, doc: Doc) -> Doc:
        """
        Slightly modified from spacy.pipeline.function.merge_entities to accommodate
        stopword trimming.
        """
        with doc.retokenize() as retokenizer:
            # Merge discovered entities. Ones found via `PipedPhraseMatcher` have label
            # "CUSTOM"
            for ent in doc.ents:
                if self.filter_entities is None or ent.label_ in self.filter_entities:
                    attrs = {
                        "tag": ent.root.tag,
                        "dep": ent.root.dep,
                        "ent_type": ent.label,
                    }

                    # need to trim leading/trailing stopwords
                    if ent.label_ != "CUSTOM" and self.stopwords is not None:
                        while ent and ent[0].lower_ in self.stopwords:
                            ent = ent[1:]
                        while ent and ent[-1].lower_ in self.stopwords:
                            ent = ent[:-1]

                    if not ent:
                        continue
                    retokenizer.merge(ent, attrs=attrs)
        return doc


def detect_phrases(
    docs_reader: Callable,
    passes: int = 1,
    lowercase: bool = False,
    detect_entities: bool = False,
    token_regex: Optional[Pattern] = None,
    min_count: int = 5,
    threshold: float = 10.0,
    max_vocab_size: float = 40_000_000,
    connector_words: Optional[Iterable[str]] = None,
    phrases: Optional[list[str]] = None,
    n_process: int = 1,
    total_docs: Optional[int] = None,
) -> dict[str, float]:
    """
    Pipeline from radimrehurek.com/gensim/models/phrases.html

    Designed to be run before preprocessing, to create a list of corpus-specific phrases

    If `connector_words` is "english", will use English connector words from gensim.
    """
    # This function is self-contained, so we defer imports here (gensim is not used
    # during preprocessing)
    from .preprocess import tokenize_docs
    from gensim.models import Phrases

    for i in range(passes):
        logger.info(f"On pass {i} of {passes}")
        doc_tokens = tokenize_docs(
            docs=docs_reader(),
            lowercase=lowercase,
            ngram_range=(1, 1),
            detect_entities=detect_entities,
            double_count_phrases=False,
            token_regex=token_regex,
            n_process=n_process,
            phrases=phrases,
            stopwords=None, # do not delete until doing connection below
        )
        phraser = Phrases(
            tqdm((toks for toks, id in doc_tokens), total=total_docs),
            min_count=min_count,
            threshold=threshold,
            max_vocab_size=max_vocab_size,
            progress_per=float("inf"),
            connector_words=frozenset(connector_words) if connector_words else [],
        )

        # for future passes
        phrases = list(phraser.export_phrases().keys())
        detect_entities = False  # these will have been added to `phrases`

    return dict(sorted(phraser.export_phrases().items(), key=lambda kv: -kv[1]))
