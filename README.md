# Is Automated Topic Model Evaluation Broken?

Code and data to run experiments for [our paper](https://arxiv.org/abs/2107.02173). Note that the [dev](https://github.com/ahoho/topics/tree/dev) branch is our "working" branch, which has a slightly improved preprocessing API and more complete instructions, although it may not be able to exactly reproduce paper results.

We're working on writing up the series of steps to reproduce the results in the paper. Please create an issue if we haven't made progress on this!

The full topic-modeling package is still a work in process. As of now, only preprocessing is handled by a unified CLI installed via `poetry` (see below).

Other steps have different requirements; models and analyses are associated with different conda environments. Running hyperparameter sweeps is also dependent on SLURM.

# Preprocessing

Preprocessing relies on [spaCy](https://spacy.io/) to efficiently tokenize text, optionally merging together detected entities and other provided phrases (e.g. `New York` -> `New_York`). This addition helps with topic readability.

## Installation

To install, you first need to get [`poetry`](https://python-poetry.org/docs/).

`poetry` can create virtual environments automatically, but will also detect any activated virtual environment and use that instead (e.g., if you are using [conda](https://docs.conda.io/en/latest/miniconda.html), as I do, run `conda create -n soup-nuts python=3.9 && conda activate soup-nuts`).

Then from the repository root, run

```console
$ poetry install --extras gensim
```

Instructions for usage can be accessed with `soup-nuts --help`

If you have trouble using poetry and need to install packages manually (meaning `soup-nuts` is no longer on your path), then you can run as a module with `python -m soup_nuts.main`.

# Model estimation

# Automated evaluation

# Human evaluation
