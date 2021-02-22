# Soup Nuts

A topic modeling package that aims to cover the entire preprocessing-estimation-analysis pipeline: it's soup to nuts.

For preprocessing, it relies on [spaCy](https://spacy.io/) to efficiently tokenize text, optionally merging together detected entities and other provided phrases (e.g. `New York` -> `New_York`). This addition has a big impact on topic readability.

**Installation**:

These instructions are temporary since this project is still in early development.

To install, you first need to get (`poetry`)[https://python-poetry.org/docs/].

`poetry` can create virtual environments automatically, but will also detect any activated virtual environment and use that instead (e.g., if you are using [`conda`](https://docs.conda.io/en/latest/miniconda.html), as I do, run `conda create -n soup-nuts python=3.9 && conda activate soup-nuts`).

Then from the repository root, run

```console
poetry install --extras gensim
```

**Usage**:

```console
$ soup-nuts [OPTIONS] COMMAND [ARGS]...
```

**Commands**:

* `Learn phrases (e.g., `New_York`) from the data.`
* `preprocess`: Preprocess documents to a document-term...
* `run-model`

## `soup-nuts preprocess`

Preprocess documents to a document-term matrix.

**Usage**:

```console
$ soup-nuts preprocess [OPTIONS] INPUT_PATH... OUTPUT_DIR
```

**Arguments**:

* `INPUT_PATH...`: File(s) containing raw text (works with glob patterns)  [required]
* `OUTPUT_DIR`: Output directory. Will save vocabulary and the document-term matrix.  [required]

**Options**:

* `--input-format [text|jsonl]`: Format of input file(s). If jsonlist, can specify an id variable  [default: text]
* `--output-format [sparse|jsonl]`: Format in which to save the document-term matrix. For `jsonl`, each row contains the document's id and word counts, e.g. {'id': '0001', 'counts': {'apple': 2, 'cake': 1}. `sparse` is a `scipy.sparse.csr.csr_matrix`, vocab & ids in separate files  [default: sparse]
* `--lines-are-documents / --no-lines-are-documents`: Treat each line in a file as a document (else, each file is a document).   [default: True]
* `--jsonl-text-key TEXT`
* `--jsonl-id-key TEXT`
* `--lowercase / --no-lowercase`: [default: False]
* `--token-regex TEXT`: How to retain tokens: `alpha`: keep anything containing at least one letter. `wordlike`: keep alphanumeric, eliminating words with all numbers, or punctuation outside of hyphens, periods, and underscores. `all`: do not filter other values will be interpreted as regex.  [default: alpha]
* `--ngram-range <INTEGER INTEGER>...`: Range of ngrams to use. e.g., (1, 1) is unigrams; (1, 2) unigrams & bigrams; (2, 2) bigrams only. Not reccomended for use with phrasing options. (borrowed from sklearn.feature_extraction.text.CountVectorizer)  [default: 1, 1]
* `--min-doc-freq FLOAT RANGE`: Ignore terms with a document frequency lower than this threshold (if < 1, treated as a proportion of documents).(borrowed from sklearn.feature_extraction.text.CountVectorizer)  [default: 1]
* `--max-doc-freq FLOAT RANGE`: Ignore terms with a document frequency higher than this threshold, i.e., corpus-specific stopwords (if <= 1, treated as a proportion of documents).(borrowed from sklearn.feature_extraction.text.CountVectorizer)  [default: 1.0]
* `--max-vocab-size INTEGER RANGE`: Maximum size of the vocabulary. If < 1, share of total vocab to keep
* `--detect-entities / --no-detect-entities`: Automatically detect entities with spaCy, `New York` -> `New_York`.   [default: True]
* `--double-count-phrases / --no-double-count-phrases`: Collocations are included alongside constituent unigrams, `New York -> `New York New_York`. Anecdotally forms more interpretable topics.  [default: True]
* `--max-doc-size INTEGER RANGE`: Maximum document size in whitespace-delimited tokens
* `--vocabulary FILE`: Use an external vocabulary list. Overrides other preprocessing options.
* `--phrases FILE`: Filepath of known phrases (e.g., `New_York`). Must be connected with underscore
* `--stopwords FILE`: Filepath of stopwords, one word per line. Set to `english` to use spaCy defaults or `none` to not remove stopwords
* `--encoding TEXT`: [default: utf8]
* `--n-process INTEGER`: [default: -1]
* `--help`: Show this message and exit.

## `soup-nuts detect-phrases`

Learn phrases (e.g., `New_York`) from the data.

**Usage**:

```console
$ soup-nuts Learn phrases (e.g., `New_York`) from the data. [OPTIONS] INPUT_PATH... OUTPUT_DIR
```

**Arguments**:

* `INPUT_PATH...`: File(s) containing raw text (works with glob patterns)  [required]
* `OUTPUT_DIR`: Output directory. Will save vocabulary file and a document-term matrix.  [required]

**Options**:

* `--input-format [text|jsonl]`: Format of input file(s). If jsonlist, can specify an id variable  [default: text]
* `--lines-are-documents / --no-lines-are-documents`: Treat each line in a file as a document (else, each file is a document).   [default: True]
* `--jsonl-text-key TEXT`
* `--jsonl-id-key TEXT`
* `--max-doc-size INTEGER RANGE`: Maximum document size in whitespace-delimited tokens
* `--passes INTEGER`: Passes over the data, where more passes leads to longer phrase detection, e.g., pass one yields `New_York`, pass two yields `New_York_City`.  [default: 1]
* `--lowercase / --no-lowercase`: [default: False]
* `--detect-entities / --no-detect-entities`: Automatically detect entities with spaCy, `New York` -> `New_York`. If you plan to set this to `True` during preprocessing, do it now and set it to `False` after  [default: True]
* `--token-regex TEXT`: How to retain tokens: `alpha`: keep anything containing at least one letter. `wordlike`: keep alphanumeric, eliminating words with all numbers, or punctuation outside of hyphens, periods, and underscores. `all`: do not filter other values will be interpreted as regex.  [default: alpha]
* `--min-count INTEGER`: [default: 5]
* `--threshold FLOAT`: [default: 10.0]
* `--max-vocab-size FLOAT`: [default: 40000000]
* `--connector-words TEXT`: Point to a path of connector words, or use `english` to use common english articles (from gensim). Set to `none` to not use. From gensim docs: 'Set of words that may be included within a phrase, without affecting its scoring. No phrase can start nor end with a connector word; a phrase may contain any number of connector words in the middle.'   [default: english]
* `--phrases FILE`: Path to list of already-known phrases (e.g., `New_York`). Must be connected with underscore.
* `--encoding TEXT`: [default: utf-8]
* `--n-process INTEGER`: [default: -1]
* `--help`: Show this message and exit.

## `soup-nuts run-model`

**Usage**:

```console
$ soup-nuts run-model [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.
