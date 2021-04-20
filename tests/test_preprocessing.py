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
        max_phrase_len: Optional[int] = None,
        token_regex: re.Pattern = None,
        min_chars: int = 0,
        lemmatize: bool = False,
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
            max_phrase_len=max_phrase_len,
            token_regex=token_regex,
            min_chars=min_chars,
            lemmatize=lemmatize,
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

        # test that we can do multiple documents
        self.assert_equal_tokenization(
            [
                "this is document one",
                "this is document two",
            ],
            [
                ["this", "is", "document", "one"],
                ["this", "is", "document", "two"],
            ]
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

        # it should never return empty lists
        self.assert_equal_tokenization(
            ["only stopwords", "not only stopwords"],
            [["not"]], 
            stopwords=["only", "stopwords"],
        )

    def test_custom_phrase_creation(self):
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
            [["this", "is", "a", "hyphenated-word", "workaround"]],
            phrases=["hyphenated-word"],
        )

    def test_max_phrase_length_filtering(self):
        # note that these are eliminated _entirely_, we don't trim down to the
        # root or anything may be a TODO
        self.assert_equal_tokenization(
            ["10 million friendly Armenian pirates"],
            [["10", "million", "friendly", "Armenian", "pirates"]],
            detect_noun_chunks=True,
            max_phrase_len=2,
        )

        # custom phrases don't get trimmed
        self.assert_equal_tokenization(
            ["10 million friendly Armenian pirates"],
            [["10_million_friendly_Armenian_pirates"]],
            phrases=["10_million_friendly_Armenian_pirates"],
            max_phrase_len=2,
        )

    def test_entity_detection(self):
        # test whether entities are found
        self.assert_equal_tokenization(
            ["I live in the United States of America"],
            [["I", "live", "in", "the_United_States_of_America"]],
            detect_entities=True,
        )

        # make sure outer stopwords are REMOVED for detected phrases,
        # but inner stopwords are retained
        self.assert_equal_tokenization(
            ["I live in the United States of America"],
            [["I", "live", "in", "United_States_of_America"]],
            detect_entities=True,
            stopwords=["the", "of"],
        )

    def test_custom_phrase_with_entity_detection(self):
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

    def test_custom_phrase_with_noun_chunk_detection(self):
        # custom phrases should take precendence over noun detection
        self.assert_equal_tokenization(
            ["Eat a red apple in the United States of America"],
            [["Eat", "a", "red_apple", "in", "the_United_States", "of", "America"]],
            detect_noun_chunks=True,
            phrases=["red_apple"],
        )

        # below is the correct behavior---note that any overlap of a detected noun
        # chunk with a custom phrase will break the noun chunk detection. there is no
        # "United_States" here
        self.assert_equal_tokenization(
            ["Eat a red apple in the United States of America"],
            [["Eat", "a_red_apple", "in_the", "United", "States", "of", "America"]],
            detect_noun_chunks=True,
            phrases=["in_the"],
        )

    def test_noun_chunk_with_entity_detection(self):
        # for combination with entity detection, the longer span is preferred
        self.assert_equal_tokenization(
            ["Eat a red apple in the United States of America"],
            [["Eat", "a_red_apple", "in", "the_United_States_of_America"]],
            detect_noun_chunks=True,
            detect_entities=True,
        )

        self.assert_equal_tokenization(
            ["N.K. Jemisin writes a great book"],
            [["N.K._Jemisin", "writes", "a_great_book"]],
            detect_noun_chunks=True,
            detect_entities=True,
        )

    def test_custom_phrase_with_noun_chunk_with_entity_detection(self):
        # for combination with entity detection, the longer span is preferred
        self.assert_equal_tokenization(
            ["Eat a red apple in the United States of America"],
            [["Eat", "a_red_apple", "in", "the_United_States", "of_America"]],
            detect_noun_chunks=True,
            detect_entities=True,
            phrases=["of_America"],
        )

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
            ["keep: a0q é é10 aé ming!blop okey_dokey hunky-dory 1000 29@44 %_3 5_5 drop: ! , "],
            [["keep", "a0q", "é", "é10", "aé", "ming!blop", "okey_dokey", "hunky", "dory", "1000", "29@44", "_", "3", "5_5", "drop"]],
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

        # hyphens
        self.assert_equal_tokenization(
            ["I feel hunky-dory"],
            [["I", "feel", "hunky-dory"]],
            token_regex=main.token_regex_callback("wordlike"),
            phrases=["hunky-dory"],
        )

        # custom regex
        self.assert_equal_tokenization(
            ["1000 a+ $999"],
            [["1000", "999"]],
            token_regex=re.compile("[0-9]"),
        )


    def test_lemmatization(self):
        self.assert_equal_tokenization(
            ["A bear eats in the woods"],
            [["a", "bear", "eat", "in", "the", "wood"]],
            lemmatize=True
        )

        # lemmatization should not apply to phrases
        self.assert_equal_tokenization(
            ["A bear eats in the woods"],
            [["A_bear", "eat", "in", "the_woods"]],
            lemmatize=True,
            detect_noun_chunks=True,
        )

    def test_vocabulary_filtering(self):
        self.assert_equal_tokenization(
            ["keep only specified vocabulary"],
            [["keep", "vocabulary"]],
            vocabulary=["keep", "vocabulary"],
        )

        self.assert_equal_tokenization(
            ["test Case Sensivity"],
            [["test"]],
            vocabulary=["test", "case", "sensitivity"],
        )

        self.assert_equal_tokenization(
            ["TEST Case Insensitivity"],
            [["test", "case",  "insensitivity"]],
            vocabulary=["TEST", "case", "insensitivity"],
            lowercase=True
        )

    def test_full_sentence(self):
        doc =  [(
            "According to the Centers for Disease Control and Prevention, 8.1 million "
            "American adults used e-cigarettes every day or some days in 2018, and about "
            "5.4 million American middle and high school students have used "
            "an e-cigarette in the last 30 days."
        )]

        # try a real-world example.
        self.assert_equal_tokenization(
            doc,
            [[
                'According', 'Centers_for_Disease_Control_and_Prevention', 
                'Centers', 'Disease', 'Control', 'Prevention', 
                'million', 'American', 'adults', 'e-cigarettes',
                'day', 'days', 'million', 'American', 'middle', 'high', 'school',
                 'students', 'e-cigarette', '30_days', 'days'
            ]],
            detect_entities=True,
            detect_noun_chunks=True,
            double_count_phrases=True,
            token_regex=main.token_regex_callback("wordlike"),
            stopwords=main.stopwords_callback("english"),
            phrases=["e-cigarette", "e-cigarettes"], # hyphens to correct tokenization
        )

        self.assert_equal_tokenization(
            doc,
            [[
                'According', 'Centers_for_Disease_Control_and_Prevention',
                '8.1_million_American_adults', 'e-cigarettes', 'day', 'days',
                '2018', '5.4', 'million', 'American', 'middle', 'high', 'school',
                'students', 'e-cigarette', '30_days'
            ]],
            detect_entities=True,
            detect_noun_chunks=True,
            double_count_phrases=False,
            max_phrase_len=8,
            token_regex=main.token_regex_callback("alphanum"),
            stopwords=main.stopwords_callback("english"),
            phrases=["e-cigarette", "e-cigarettes"], # hyphens to correct tokenization
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