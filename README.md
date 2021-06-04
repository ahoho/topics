# Soup Nuts

A topic modeling package that aims to cover the entire preprocessing-estimation-analysis pipeline: it's soup to nuts.

For preprocessing, it relies on [spaCy](https://spacy.io/) to efficiently tokenize text, optionally merging together detected entities and other provided phrases (e.g. `New York` -> `New_York`). This addition has a big impact on topic readability.

## Installation

These instructions are temporary since this project is still in early development.

To install, you first need to get [`poetry`](https://python-poetry.org/docs/).

`poetry` can create virtual environments automatically, but will also detect any activated virtual environment and use that instead (e.g., if you are using [conda](https://docs.conda.io/en/latest/miniconda.html), as I do, run `conda create -n soup-nuts python=3.9 && conda activate soup-nuts`).

Then from the repository root, run

```console
$ poetry install --extras gensim
```

Finally, you will need to download the spaCy models with

```console
$ python -m spacy download en_core_web_sm
```

Instructions for usage can be accessed with `soup-nuts --help`
