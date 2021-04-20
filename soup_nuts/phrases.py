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
        "case_sensitive": True,
    },
)
def make_phrase_matcher(
    nlp: Language,
    name: str,
    *,
    phrases: Iterable[str],
    case_sensitive: bool = True,
):
    return PipedPhraseMatcher(nlp, phrases, case_sensitive)


class PipedPhraseMatcher:
    def __init__(
        self,
        nlp: Language,
        phrases: Iterable[str],
        case_sensitive: bool = True,
    ):
        """
        Basically a "custom" NER pipeline relying on a predefined phraselist.

        If these are to take priority (in case of span overlap), run before standard NER.

        `case_sensitive` determines whether we lowercase
        when checking for membership in `phrases`

        Borrowed from spacy.io/usage/rule-based-matching#phrasematcher
        """
        self.matcher = PhraseMatcher(nlp.vocab, attr="ORTH" if case_sensitive else "LOWER")
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
        "max_phrase_len": None,
    },
    retokenizes=True,
)
def make_phrase_merger(
    nlp: Language,
    name: str,
    *,
    stopwords: Optional[list[str]] = None,
    filter_entities: Optional[list[str]] = None,
    max_phrase_len: Optional[int] = None,
):
    return PhraseMerger(stopwords, filter_entities, max_phrase_len)


class PhraseMerger:
    """
    Merge together detected entities and noun phrases discovered earlier in the pipeline

    Either `ner`, `match_phrases`, or [`tagger`, `parser`] (for noun chunks) required.
    """
    def __init__(
        self,
        stopwords: Optional[list[str]] = None,
        filter_entities: Optional[list[str]] = None,
        max_phrase_len: Optional[int] = None,
    ):
        """
        Stopwords and entity labels are preferred as sets since lookup is O(1).
        """
        self.stopwords = set(stopwords) if stopwords else None
        self.filter_entities = set(filter_entities) if filter_entities else None
        self.max_phrase_len = max_phrase_len or float("inf")

    def __call__(self, doc: Doc) -> Doc:
        """
        Slightly modified from spacy.pipeline.function.merge_entities to accommodate
        stopword trimming.
        """
        with doc.retokenize() as retokenizer:
            # Merge discovered entities / noun chunks. 
            # Ones found via `PipedPhraseMatcher` have label "CUSTOM"
            ents = [
                ent for ent in doc.ents
                if self.filter_entities is None or ent.label_ in self.filter_entities
            ]
            custom = set(tok.i for ent in ents for tok in ent if ent.label_ == "CUSTOM")

            noun_chunks = []
            if doc.has_annotation("DEP"):
                # ensure precedence of CUSTOM phrases
                noun_chunks = [
                    noun for noun in doc.noun_chunks
                    if not any(tok.i in custom for tok in noun)
                ]
            
            # eliminate overlapping spans, keeping the longest
            # NB that, given earlier filtering, CUSTOM phrases should never be subsumed/
            # broken up
            phrases = filter_spans([
                p for p in ents + noun_chunks
                if p.label_ == "CUSTOM" or len(p) <= self.max_phrase_len
            ])

            for phrase in phrases:
                attrs = {
                    "tag": phrase.root.tag,
                    "dep": phrase.root.dep,
                    "ent_type": phrase.label,
                }
                # need to trim leading/trailing stopwords
                if phrase.label_ != "CUSTOM" and self.stopwords is not None:
                    while phrase and phrase[0].lower_ in self.stopwords:
                        phrase = phrase[1:]
                    while phrase and phrase[-1].lower_ in self.stopwords:
                        phrase = phrase[:-1]

                if not phrase:
                    continue

                retokenizer.merge(phrase, attrs=attrs)

        return doc


def detect_phrases(
    docs_reader: Callable,
    passes: int = 1,
    lowercase: bool = False,
    detect_entities: bool = False,
    detect_noun_chunks: bool = False,
    token_regex: Optional[Pattern] = None,
    min_count: int = 5,
    threshold: float = 10.0,
    max_vocab_size: float = 40_000_000,
    connector_words: Optional[Iterable[str]] = None,
    phrases: Optional[Iterable[str]] = None,
    max_phrase_len: Optional[int] = None,
    n_process: int = 1,
    total_docs: Optional[int] = None,
) -> list[str]:
    """
    Pipeline from radimrehurek.com/gensim/models/phrases.html

    Designed to be run before preprocessing, to create a list of corpus-specific phrases
    """
    # This function is self-contained, so we defer imports (gensim is not used
    # during preprocessing, so preferable to keep it optional)
    from .preprocess import tokenize_docs, create_pipeline
    from gensim.models import Phrases

    connector_words = frozenset(connector_words) if connector_words else frozenset()

    for i in range(passes):
        logger.info(f"On pass {i} of {passes}")

        # reinitialize the generator and spacy model
        docs = docs_reader()
        nlp = create_pipeline(
            model_name="en_core_web_sm",
            detect_entities=detect_entities,
            detect_noun_chunks=detect_noun_chunks,
            case_sensitive=not lowercase,
            phrases=phrases,
            phrase_stopwords=connector_words,
        )

        doc_tokens = tokenize_docs(
            docs=docs,
            spacy_model=nlp,
            lowercase=lowercase,
            ngram_range=(1, 1),
            double_count_phrases=False,
            token_regex=token_regex,
            n_process=n_process,
            stopwords=None, # do not delete until doing connection below
        )
        
        phraser = Phrases(
            tqdm((toks for toks, id in doc_tokens), total=total_docs),
            delimiter="~",
            min_count=min_count,
            threshold=threshold,
            max_vocab_size=max_vocab_size,
            progress_per=float("inf"),
            connector_words=connector_words,
        )

        # for future passes
        entities_and_noun_chunks = [w for w in phraser.vocab if '_' in w and '~' not in w]
        detected_phrases = [w.replace("~", "_") for w in phraser.export_phrases()]
        phrases = detected_phrases + entities_and_noun_chunks
        detect_entities = False  # these will have been added to `phrases`
        detect_noun_chunks = False

    if max_phrase_len:
        phrases = [p for p in phrases if p.count("_") + 1 <= max_phrase_len]

    return list(set(phrases))
