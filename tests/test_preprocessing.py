import unittest
from typing import Union

from soup_nuts import preprocess


class TestDataReading(unittest.TestCase):
    def test_line_counting(self):
        pass

    def test_raw_data_read(self):
        pass

    def test_jsonl_read(self):
        pass


class TestTokenization(unittest.TestCase):
    default_kwargs = {
        "lowercase": False,
        "as_tuples": False,
        "ngram_range": (1, 1),
        "remove_stopwords": False,
        "detect_entities": False,
        "double_count_phrases": False,
        "token_regex": None,
        "vocabulary": None,
        "phrases": None,
        "stopwords": None,
        "n_process": 1,
    }

    def assert_equal_tokenization(
        self,
        docs: list[str],
        compare: list[str],
        **kwargs,
    ):
        new_kwargs = self.default_kwargs
        new_kwargs.update(**kwargs)
        tokenized = list(preprocess.tokenize_docs(docs, **new_kwargs))
        self.assertEqual(tokenized, compare)

    def test_basic_tokenization(self):
        # test without ids
        self.assert_equal_tokenization(
            ["This is a sentence."],
            [["This", "is", "a", "sentence", "."]],
        )

        # test that ids are retained
        self.assert_equal_tokenization(
            [("This is a sentence.", "id")],
            [(["This", "is", "a", "sentence", "."], "id")],
            as_tuples=True,
        )

    def test_lowercasing(self):
        self.assert_equal_tokenization(
            ["This is a grEAt sentence."],
            [["this", "is", "a", "great", "sentence", "."]],
            lowercase=False,
        )

    def test_ngram_creation(self):
        pass

    def test_stopword_removal(self):
        pass

    def test_entity_detection(self):
        pass

    def test_phrase_double_counting(self):
        pass

    def test_token_regex_filtering(self):
        pass

    def test_vocabulary_filtering(self):
        pass

    def test_phrase_merging(self):
        pass


class TestDocumentTermMatrix(unittest.TestCase):
    def test_matrix_creation(self):
        pass

    def test_min_doc_freq_filtering(self):
        pass

    def test_max_doc_freq_filtering(self):
        pass

    def test_max_vocab_filtering(self):
        pass