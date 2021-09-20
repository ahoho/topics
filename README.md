# Soup-to-Nuts Topic Modeling

We introduce a topic modeling package that aims to cover the entire preprocessing-estimation-analysis pipeline.

The package is still a work in process. As of now, only preprocessing is handled by a unified CLI installed via `poetry` (see below).

Other steps have different requirements; models and analyses are associated with different conda environments. Running hyperparameter sweeps is also dependent on SLURM.

# Installation

These instructions are temporary since this project is still in early development.

To install, you first need to get [`poetry`](https://python-poetry.org/docs/).

`poetry` can create virtual environments automatically, but will also detect any activated virtual environment and use that instead (e.g., if you are using [conda](https://docs.conda.io/en/latest/miniconda.html), run `conda create -n soup-nuts python=3.9 && conda activate soup-nuts`).

Then from the repository root, run

```console
$ poetry install --extras gensim
```

As stated above, models need their own environments. Requirements are in the `.yml` files in each of the `soup_nuts/models/<model_name>` directories, and can be installed with

```console
conda env create -f <environment_name>.yml
```

(names for each are set in the first line of each `yml` file, and can be changed as needed)

# Preprocessing

Preprocessing relies on [spaCy](https://spacy.io/) to efficiently tokenize text, optionally merging together detected entities and other provided phrases (e.g. `New York` -> `New_York`). This addition helps with topic readability.

## Running preprocessing

Instructions for usage can be accessed with `soup-nuts preprocess --help`. Some models and settings are not yet fully integrated in the pipeline, and require additional steps or specific flags, as described below (NB: they also introduce some redundancy in the data. If you will only be using `scholar` and are short on space, feel free to delete intermediate files)

- **`scholar`** model
    - For `soup-nuts preprocess`, use these flags: `--output-format jsonl --retain-text`
    - `scholar` requires a specific input format. Run the python script `data/convert_processed_data_to_scholar_format.py` (dependencies are in `soup_nuts/models/scholar/scholar.yml`).
- **Covariates/labels** (currently only supported in `scholar`)
    - For `soup-nuts preprocess`, specify labels/covariates with these flags: `--input-format jsonl --jsonl-text-key <your_text_key> --output-format jsonl --jsonl-metadata-keys <key1,key2,key3,...>` (in addition to steps for `scholar`)
- **Knowledge distillation** (currently only supported in `scholar`)
    - For `soup-nuts preprocess`, retain the text as "metadata" with these flags `--input-format jsonl --jsonl-text-key <your_text_key> --output-format jsonl --jsonl-metadata-keys <your_text_key>` (in addition to steps for `scholar`)


# Models

All models currently require independent conda environments. Although some effort has been made to unify the arguments of the models, for now they should be treated separately.

Running **knowledge distillation** also requires a separate environment, as it involves the use of the [`transformers`](https://github.com/huggingface/transformers) library.
