import io
import re
from collections.abc import Iterable
from typing import Optional, Union

from spacy import Language

from soup_nuts import preprocess, main


class TestTokenization:
    def assert_equal_tokenization(
        self,
        docs: list[str],
        compare: list[str],
        spacy_model: Union[str, Language] = "en_core_web_sm",
        lowercase: bool = False,
        as_tuples: bool = False,
        ngram_range: tuple[int, int] = (1, 1),
        detect_entities: bool = False,
        detect_noun_chunks: bool = False,
        double_count_phrases: bool = False,
        token_regex: re.Pattern = None,
        vocabulary: Optional[Iterable[str]] = None,
        phrases: Optional[Iterable[str]] = None,
        stopwords: Optional[Iterable[str]]  = None,
        n_process: int = 1,

    ):
        tokenized = list(preprocess.tokenize_docs(
            docs,
            spacy_model=spacy_model,
            lowercase=lowercase,
            as_tuples=as_tuples,
            ngram_range=ngram_range,
            detect_entities=detect_entities,
            detect_noun_chunks=detect_noun_chunks,
            double_count_phrases=double_count_phrases,
            token_regex=token_regex,
            vocabulary=vocabulary,
            phrases=phrases,
            stopwords=stopwords,
            n_process=n_process,
        ))
        tokenized
        assert tokenized == compare

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
            lowercase=True,
        )

    def test_ngram_creation(self):
        self.assert_equal_tokenization(
            ["one two three four five six seven"],
            [[
                "one", "two", "three", "four", "five", "six", "seven",
                "one_two", "two_three", "three_four", "four_five", "five_six", "six_seven",
                "one_two_three", "two_three_four", "three_four_five", "four_five_six", "five_six_seven",
            ]],
            ngram_range=(1, 3),
        )

        self.assert_equal_tokenization(
            ["one two three four five six seven"],
            [[
                "one_two", "two_three", "three_four", "four_five", "five_six", "six_seven",
            ]],
            ngram_range=(2, 2),
        )

        self.assert_equal_tokenization(
            ["one two three four five six seven"],
            [[
                "one_two_three", "two_three_four", "three_four_five", "four_five_six", "five_six_seven",
            ]],
            ngram_range=(3, 3),
        )

    def test_stopword_removal(self):
        self.assert_equal_tokenization(
            ["let's see this sentence MY_STOPWORD myotherstopword without custom stopwords"],
            [["let", "'s", "see", "this", "sentence", "without", "custom", "stopwords"]],
            stopwords=["MY_STOPWORD", "myotherstopword"],
        )

        # make sure that if stopwords are _NOT_ specified, then they aren't used
        self.assert_equal_tokenization(
            ["a the boy"],
            [["a", "the", "boy"]],
            stopwords=[],
        )

    def test_phrase_merging(self):
        # make sure a custom phrase merging works
        self.assert_equal_tokenization(
            ["make sure this custom phrase is found"],
            [["make", "sure", "this", "custom_phrase", "is", "found"]],
            phrases=["custom_phrase"],
        )

        # make sure stopwords are RETAINED for custom phrases
        self.assert_equal_tokenization(
            ["make sure this custom phrase is found in this sentence"],
            [["make", "sure", "this_custom_phrase", "is", "found", "in", "sentence"]],
            phrases=["this_custom_phrase"],
            stopwords=["this"],
        )

        self.assert_equal_tokenization(
            ["this is a hyphenated-word workaround"],
            [["this", "is", "a", "hyphenated_word", "workaround"]],
            phrases=["hyphenated_word"],
        )

    def test_entity_detection(self):
        # test whether entities are found
        self.assert_equal_tokenization(
            ["I live in the United States of America"],
            [["I", "live", "in", "the_United_States_of_America"]],
            detect_entities=True,
        )

        # make sure outer stopwords are REMOVED for detected phrases
        self.assert_equal_tokenization(
            ["I live in the United States of America"],
            [["I", "live", "in", "United_States_of_America"]],
            detect_entities=True,
            stopwords=["the"],
        )

        # make sure custom phrases take precedence over entity detection
        self.assert_equal_tokenization(
            ["I live in the United States of America"],
            [["I", "live", "in", "United_States", "of_America"]],
            detect_entities=True,
            stopwords=["of", "the"],
            phrases=["of_America"],
        )

        self.assert_equal_tokenization(
            ["I live in the United States of America"],
            [["I", "live", "in", "United", "States_of", "America"]],
            detect_entities=True,
            stopwords=["of", "the"],
            phrases=["States_of"],
        )

    def test_noun_chunk_detection(self):
        # test whether chunks are found
        self.assert_equal_tokenization(
            ["Eat a red apple in the United States of America"],
            [["Eat", "a_red_apple", "in", "the_United_States", "of", "America"]],
            detect_noun_chunks=True,
        )

        # make sure outer stopwords are REMOVED for detected phrases
        self.assert_equal_tokenization(
            ["Eat a red apple in the United States of America"],
            [["Eat", "red_apple", "in", "United_States", "of", "America"]],
            detect_noun_chunks=True,
            stopwords=["a", "the"],
        )
        # TODO: 
        # - interaction with custom phrases
        # - interaction with entity detection
        # - interaction of all three

    def test_phrase_double_counting(self):
        self.assert_equal_tokenization(
            ["make sure this custom phrase is found"],
            [["make", "sure", "this", "custom_phrase", "custom", "phrase", "is", "found"]],
            phrases=["custom_phrase"],
            double_count_phrases=True,
        )

        # double counted components should be filtered separately
        # test stopword filtering
        self.assert_equal_tokenization(
            ["make sure this custom phrase is found"],
            [["make", "sure", "this", "custom_phrase", "custom", "is", "found"]],
            phrases=["custom_phrase"],
            double_count_phrases=True,
            stopwords=["phrase"],
        )

        # test regex filtering
        self.assert_equal_tokenization(
            ["make sure this custom phrase is found"],
            [["custom_phrase"]],
            phrases=["custom_phrase"],
            double_count_phrases=True,
            token_regex=re.compile("_"),
        )

        # test vocab filtering
        self.assert_equal_tokenization(
            ["make sure this custom phrase is found"],
            [["custom_phrase", "custom"]],
            phrases=["custom_phrase"],
            double_count_phrases=True,
            vocabulary=["custom"],
        )


    def test_token_regex_filtering(self):
        self.assert_equal_tokenization(
            ["keep: a0q é é10 aé ming!blop okey_dokey hunky-dory drop: 1000 ! , 29@44 %_3 5_5"],
            [["keep", "a0q", "aé", "ming!blop", "okey_dokey", "hunky", "dory", "drop"]],
            token_regex=main.token_regex_callback("alpha"),
        )

        self.assert_equal_tokenization(
            ["keep: a0q é é10 aé ming!blop okey_dokey hunky-dory drop: 1000 ! , 29@44 %_3 5_5"],
            [["keep", "a0q", "é", "é10", "aé", "ming!blop", "okey_dokey", "hunky", "dory", "drop"]],
            token_regex=main.token_regex_callback("alphanum"),
        )

        self.assert_equal_tokenization(
            ["keep: a0q é é10 aé okey_dokey hunky-dory drop: 1000 ! , 29@44 %_3 5_5 $1,000 ming!blop"],
            [["keep", "a0q", "aé", "okey_dokey", "hunky", "dory", "drop"]],
            token_regex=main.token_regex_callback("wordlike"),
        )

        self.assert_equal_tokenization(
            ["keep: a0q é10 okey_dokey hunky-dory 1000 ! , 29@44 %_3 $1,000 ming!blop"],
            [[
                "keep", ":", "a0q", "é10", "okey_dokey", "hunky", "-", "dory",
                "1000", "!", ",", "29@44", "%", "_", "3", "$", "1,000", "ming!blop"
            ]],
            token_regex=main.token_regex_callback("all"),
        )

        # custom regex
        self.assert_equal_tokenization(
            ["1000 a+ $999"],
            [["1000", "999"]],
            token_regex=re.compile("[0-9]"),
        )

    def test_vocabulary_filtering(self):
        self.assert_equal_tokenization(
            ["keep only specified vocabulary"],
            [["keep", "vocabulary"]],
            vocabulary=["keep", "vocabulary"],
        )

        self.assert_equal_tokenization(
            ["test Case Sensivity"],
            [[]],
            vocabulary=["case", "sensitivity"],
        )

        self.assert_equal_tokenization(
            ["TEST Case Insensitivity"],
            [["test", "case",  "insensitivity"]],
            vocabulary=["TEST", "case", "insensitivity"],
            lowercase=True
        )


# TODO:
# class TestDocumentTermMatrix:
#     def test_matrix_creation(self):
#         pass

#     def test_min_doc_freq_filtering(self):
#         pass

#     def test_max_doc_freq_filtering(self):
#         pass

#     def test_max_vocab_filtering(self):
#         pass


# class TestDataReading:
#     def test_line_counting(self):
#         pass

#     def test_raw_data_read(self):
#         pass

#     def test_jsonl_read(self):
#         pass