# Is Automated Topic Model Evaluation Broken?

Code and data to run experiments 

We introduce topic modeling package that aims to cover the entire preprocessing-estimation-analysis pipeline.

The package is still a work in process. As of now, only preprocessing is handled by a unified CLI installed via `poetry` (see below).

Other steps have different requirements; models and analyses are associated with different conda environments. Running hyperparameter sweeps is also dependent on SLURM.

# Preprocessing

Preprocessing relies on [spaCy](https://spacy.io/) to efficiently tokenize text, optionally merging together detected entities and other provided phrases (e.g. `New York` -> `New_York`). This addition helps with topic readability.

## Installation

These instructions are temporary since this project is still in early development.

To install, you first need to get [`poetry`](https://python-poetry.org/docs/).

`poetry` can create virtual environments automatically, but will also detect any activated virtual environment and use that instead (e.g., if you are using [conda](https://docs.conda.io/en/latest/miniconda.html), as I do, run `conda create -n soup-nuts python=3.9 && conda activate soup-nuts`).

Then from the repository root, run

```console
$ poetry install --extras gensim
```

Instructions for usage can be accessed with `soup-nuts --help`

# Model estimation



# Automated evaluation

# Human evaluation