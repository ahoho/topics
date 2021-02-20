from collections.abc import Iterable
from typing import Optional

from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans
from spacy.lang.en.stop_words import STOP_WORDS

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
        ents = [Span(doc, start=s, end=e, label="CUSTOM") for _, s, e in self.matcher(doc)]
        ents = filter_spans(ents)
        doc.set_ents(ents)
        return doc


@Language.factory(
    "merge_phrases",
    default_config={
        "stopwords": STOP_WORDS,
        "filter_entities": {'PERSON', 'FACILITY', 'GPE', 'LOC', 'CUSTOM'},
    },
    retokenizes=True,
)
def make_phrase_merger(
    nlp: Language,
    name: str,
    *,
    stopwords: Optional[set[str]] = None,
    filter_entities: Optional[set[str]] = None,
):
    return PhraseMerger(stopwords, filter_entities)


class PhraseMerger:
    def __init__(
        self,
        stopwords: Optional[set[str]] = None,
        filter_entities: Optional[str] = None,
    ):
        """
        Stopwords, and entity labels are preferred as sets since lookup is O(1).
        """
        self.stopwords = stopwords
        self.filter_entities = filter_entities

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
                        "tag": ent.root.tag, "dep": ent.root.dep, "ent_type": ent.label
                    }

                    # need to trim leading/trailing stopwords
                    if ent.label_ != "CUSTOM" and self.stopwords is not None:
                        while ent[0].lower_ in self.stopwords:
                            ent = ent[1:]
                        while ent[-1].lower_ in self.stopwords:
                            ent = ent[:-1]
                    
                    if not ent:
                        continue
                    retokenizer.merge(ent, attrs=attrs)
        return doc



def detect_phrases(
    docs: Iterable[str, str],
    lowercase: bool = False,
    max_phrase_length: int = 3,
    min_count: int = 5,
    threshold: float = 10.0,
    max_vocab_size: float = 40_000_000,
    connector_words: Optional[set[str]] = None,
) -> list[str]:
    """
    TODO:
        Implement the pipeline from https://radimrehurek.com/gensim/models/phrases.html
        Designed to be run before preprocessing, to create a list of corpus-specific
        phrases
    """
    nlp = spacy.load(
        "en_core_web_sm",
        exclude=["lemmatizer", "attribute_ruler", "ner", ""]
    )
    tokens = tokenize_docs(
        docs,

    )
    raise NotImplementedError("TODO")