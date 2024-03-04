# Topic Model Evaluation

Package and data to run experiments for our NeurIPS paper, [*Is Automated Topic Model Evaluation Broken?*](https://arxiv.org/abs/2107.02173) and our Findings of EMNLP paper, [*Are Neural Topic Models Broken?*](https://aclanthology.org/2022.findings-emnlp.390/)

Preprocessing & coherence calculations are provided as part of an easy-to-use, well-documented package called `soup-nuts` (see installation and usage instructions below). Links to the processed Wikipedia data used in the paper are also listed below. We hope that this tool encourages standardized & reproducible topic model evaluation.

Data is linked for download [below](#download-data)

Please cite us if you find this package useful, and do not hesitate to create an issue or email us if you have problems!

If you use the human annotations or preprocessing:
```
@inproceedings{hoyle-etal-2021-automated,
    title = "Is Automated Topic Evaluation Broken? The Incoherence of Coherence",
    author = "Hoyle, Alexander Miserlis  and
      Goel, Pranav  and
      Hian-Cheong, Andrew and
      Peskov, Denis and
      Boyd-Graber, Jordan and
      Resnik, Philip",
    booktitle = "Advances in Neural Information Processing Systems",
    year = "2021",
    url = "https://arxiv.org/abs/2107.02173",
}
```

If you evaluate ground-truth evaluations or stability:
```
@inproceedings{hoyle-etal-2022-neural,
    title = "Are Neural Topic Models Broken?",
    author = "Hoyle, Alexander Miserlis  and
      Goel, Pranav  and
      Sarkar, Rupak  and
      Resnik, Philip",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.390",
    doi = "10.18653/v1/2022.findings-emnlp.390",
    pages = "5321--5344",
}
```

# Installation
## Preprocessing and metrics
To install the preprocessing and metric evaluation package (called `soup-nuts`), you first need to get [`poetry`](https://python-poetry.org/docs/).

`poetry` can create virtual environments automatically, but will also detect any activated virtual environment and use that instead (e.g., if you are using [conda](https://docs.conda.io/en/latest/miniconda.html), run `conda create -n soup-nuts python=3.9 && conda activate soup-nuts`).

Then from the repository root, run

```console
$ poetry install
```

Check the installation with 

```console
$ soup-nuts --help
```

If you do not use poetry, or you have issues with installation, you can run with `python -m soup_nuts.main <command name>`

## Models
Models need their own environments. Requirements are in the `.yml` files in each of the `soup_nuts/models/<model_name>` directories, and can be installed with

```console
$ conda env create -f <environment_name>.yml
```

(names for each are set in the first line of each `yml` file, and can be changed as needed)

# Use 
## Preprocessing

Preprocessing relies on [spaCy](https://spacy.io/) to efficiently tokenize text, optionally merging together detected entities and other provided phrases (e.g. `New York` -> `New_York`). This addition helps with topic readability.

Thorough instructions for usage can be accessed with 

```console
$ soup-nuts preprocess --help
```

Below we list a partial list of arguments (again, see `--help` for more):
 - Preprocessing:
    - `--lowercase`, `--ngram-range`, `--min-doc-freq`, `--max-doc-freq`, `--max-vocab-size`
        - Standard preprocessing arguments with semantics borrowed from [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) in sklearn
    - `--limit-vocab-by-df`
        - If setting `max-vocab-size`, sort the terms by their document-frequency rather than the overall term frequency
    - `--detect-entities`, `--detect-noun-chunks`
        - Detect entities (`'New York'->'New_York'`) and noun chunks (`'8.1 million American adults'-> '8.1_million_American_adults'`) with spaCy. The latter is a bit time-intensive and can lead to vocabulary size exposions.
    - `--double-count-phrases`
        - Collocations are included alongside constituent unigrams, `'New York' -> 'New York New_York'`. Per [Philip Resnik](http://users.umiacs.umd.edu/~resnik/), this helps with topic readability, although we have not tested it empirically.
    - `--lemmatize`
        - Topic modeling experts are fairly unified against stemming (ant to a lesser extent lemmatization), and there is [empirical work](https://mimno.infosci.cornell.edu/papers/schofield_tacl_2016.pdf) to back it up, but we include it as an option anyway. Lemmatization is also fallible, spaCy turns `taxes` to `taxis`
    - `--vocabulary`
        - An external vocabulary list will override other preprocessing and restrict it to the provided terms.
    - `--phrases`
        - An external underscore-connected phrase list _supplements_ phrases found with spaCy (e.g., `nintendo_gameboy_advance`). Use `soup-nuts detect-phrases` to uncover statistical collocations.
 - Data formatting
    - `--text-key, --id-key`
        - The keys corresponding to the text and id in a csv or jsonlines file. ids created automatically based on the line number in the file if not provided.
    - `--metadata-keys`
        - Keys in csv or jsonlines input data that you would like to pass through to the processed result (in a `<split>.metadata.jsonl` file). Separate with commas, e.g., `key1,key2,key3`. Can be helpful (if storage-intensive) to include the original raw text.
    - `--output-text`
        - Output the processed text in the same order as the input (e.g., `"The lamb lies down on Broadway" -> "lamb lies down broadway"`). Needed for accurate internal coherence calculations (evaluating on a train/val/test set).

## Reproducibility
### Data

Scripts to process the data as in the NeurIPS paper:
 - [wikipedia](data/wikitext/process-data.sh)
 - [nytimes](data/nytimes/process-data.sh)

To process a new dataset in the same way, use the following setup

```console
soup-nuts preprocess \
    <your_input_file> \
    processed/${OUTPUT_DIR} \
    --input-format text \
    --lines-are-documents \
    --output-text \
    --lowercase \
    --min-doc-size 5 \
    --max-vocab-size 30000 \
    --limit_vocab_by_df \
    --max-doc-freq 0.9 \
    --detect-entities \
    --token-regex wordlike \
    --no-double-count-phrases \
    --min-chars 2
```

To use the exact vocabulary from our wikipedia settings, pass `--vocabulary` and include [this file.](https://umd-clip-public.s3.amazonaws.com/topics_neurips_2021/wikitext/vocab.json)

### Download data

We share the data for our papers here:
- NeurIPS paper data (not labeled)
    - Wikitext. `train` is the 28-thousand article [WikiText-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/), `full` is a 4.6-million article Wikipedia dump from the same period.
        - jsonlines files of fully pre-processed, sequential text to calculate coherence scores
            - Format is `{"tokenized_text": "This is a document.", "id": 0 }`
            - [train.metadata.jsonl](https://umd-clip-public.s3.amazonaws.com/topics_neurips_2021/wikitext/train.metadata.jsonl)
            - [full.metadata.jsonl](https://umd-clip-public.s3.amazonaws.com/topics_neurips_2021/wikitext/full.metadata.jsonl)
        - document-term matrices (read with `scipy.sparse.load_npz`)
            - [train.dtm.npz](https://umd-clip-public.s3.amazonaws.com/topics_neurips_2021/wikitext/train.dtm.npz)
            - [full.dtm.npz](https://umd-clip-public.s3.amazonaws.com/topics_neurips_2021/wikitext/full.dtm.npz)
         - Vocabulary file (this is not necessary)
            - [vocab.json](https://umd-clip-public.s3.amazonaws.com/topics_neurips_2021/wikitext/vocab.json)
    - NYTimes data is licenced by LDC, but please contact us and we can arrange access to processed data
- Findings of EMNLP paper (with hierarchical labels for each document). Data has `"tokenized_text"` which corresponds to the tokenized text used for the models. The topline results in the paper are from `train`; `test` is an additional setting that includes unseen labels at the lower level of the hierarchy (labels are maintained at the top level).
    -  Wikitext. Raw text is in `"text"`, labels are in `"supercategory", "category", "subcategory"` (only the last two are used in the paper)
        -  [train.metadata.jsonl](https://umd-clip-public.s3.amazonaws.com/topics-emnlp.findings-2022/wikitext-labeled/train.metadata.jsonl)
        -  [test.metadata.jsonl](https://umd-clip-public.s3.amazonaws.com/topics-emnlp.findings-2022/wikitext-labeled/test.metadata.jsonl)
    -  Bills. Raw text is in `"summary"`, labels are `"topic", "subtopic"`
        -  [train.metadata.jsonl](https://umd-clip-public.s3.amazonaws.com/topics-emnlp.findings-2022/bills-labeled/train.metadata.jsonl)
        -  [test.metadata.jsonl](https://umd-clip-public.s3.amazonaws.com/topics-emnlp.findings-2022/bills-labeled/test.metadata.jsonl)

### Metrics

We use gensim to standardize metric calculations. You can download processed reference wikipedia corpora used in the paper at the following links:

To obtain metrics on topics, run `soup-nuts coherence` with the following arguments:
 - `--topics-fpath`
    - A text file, where each line contains the words for a given topic, ordered by descending word probability (not necessary to have the full vocabulary)
 - `--reference-fpath`
    - The file used to calculate co-occurence counts. It is either a jsonl or text file, where each line has space-delimited, processed text _in the same order (sequence) as the original data_, e.g., `"The lamb lies down on Broadway" -> "lamb lies down broadway"`.
    - This is what is produced with the `--output-text` flag in `soup-nuts preprocess` (If a jsonl file is provided, it assumes the key is `"tokenized-text"`)
 - `--vocabulary-fpath`
    - The _training set_ vocabulary, that is, the vocabulary that would have been used by the model to generate the topics file being evaluated. Can either be json list/dict (if keys are terms), or a plain text file.
 - `--coherence-measure`, one of `'u_mass', 'c_v', 'c_uci', 'c_npmi'`, see [gensim](https://radimrehurek.com/gensim/models/coherencemodel.html) for details
 - `--top-n`, the number of words from each topic used to calculate coherence
 - `--window-size`, the size of the sliding window over the corpus to obtain co-occurrence counts. Leave blank to use the gensim default.
 - `--output-fpath`
    - Where you would like to save the file (e.g., model_directory/coherences.json)

As an example:

```console
soup-nuts coherence \
    <path-to-topics.txt> \
    --output-fpath ./coherences.json \
    --reference-fpath data/wikitext/processed/train.metadata.jsonl \
    --coherence-measure c_npmi \
    --vocabulary-fpath <path-to-train-vocab.json> \
    --top-n 15
```

Use `--update` to add to an existing file.

## Models

All models currently require independent conda environments. To get model code, you need to clone with the `--recurse-submodules` flag.

Although some effort has been made to unify the arguments of the models, for now they should be treated separately.

Running knowledge distillation also requires a separate environment, as it involves the use of the [`transformers`](https://github.com/huggingface/transformers) library.


### Model-specific notes

Some models and settings are not yet fully integrated in the pipeline, and require additional steps or specific flags, as described below (NB: they also introduce some redundancy in the data.)

- **`scholar`** model
    - For `soup-nuts preprocess`, use these flags: `--retain-text`
    - `scholar` requires a specific input format. Run the python script `data/convert_processed_data_to_scholar_format.py` (dependencies are in `soup_nuts/models/scholar/scholar.yml`).
- **Covariates/labels** (currently only supported in `scholar`)
    - For `soup-nuts preprocess`, specify labels/covariates with these flags: `--input-format jsonl --jsonl-text-key <your_text_key> --output-format jsonl --jsonl-metadata-keys <key1,key2,key3,...>` (in addition to steps for `scholar`)
- **Knowledge distillation** (currently only supported in `scholar`)
    - For `soup-nuts preprocess`, retain the text as "metadata" with these flags `--input-format jsonl --jsonl-text-key <your_text_key> --output-format jsonl --jsonl-metadata-keys <your_text_key>` (in addition to steps for `scholar`)


# End-to-end example
Below, we outline how to run a single mallet model on some example data.

After installing `poetry` and miniconda (see above), clone the repo, create the environment and install the packages.

```console
$ git clone -b dev https://github.com/ahoho/topics.git --recurse-submodules
$ conda create -n soup-nuts python=3.9
$ conda activate soup-nuts
$ poetry install
$ pip install pandas
```

With it installed, you can now process data. To process our example data with some sensible settings:

```console
soup-nuts preprocess \
    data/examples/speeches_2020.01.04-2020.05.04.jsonl \
    data/examples/processed-speeches \
    --input-format jsonl \
    --jsonl-text-key text \
    --jsonl-id-key id \
    --lowercase \
    --token-regex wordlike \
    --min-chars 2 \
    --min-doc-freq 5 \
    --max-doc-freq 0.95 \
    --detect-entities
```

Now that the data is processed, we can run a topic model. Let's set up mallet. You will need to download it [here](http://mallet.cs.umass.edu/download.php), then extract it:

```console
$ curl http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
$ tar -xzvf mallet-2.0.8.tar.gz
```

Then we need to create a new python environment to run it:

```console
$ cd soup_nuts/models/gensim
$ conda create -n mallet python=3.8
$ conda activate mallet
$ pip install -r requirements.txt
```


Finally, from the top-level directory, we run the model with 

```console
python soup_nuts/models/gensim/lda.py \
    --input_dir data/examples/processed-speeches \
    --output_dir results/mallet-speeches \
    --eval_path train.dtm.npz \
    --num_topics 50 \
    --mallet_path soup_nuts/models/gensim/mallet-2.0.8/bin/mallet \
    --optimize_interval 10
```

*Installation Notes: Make sure the above mallet_path points to where you installed mallet!  Otherwise you will get "returned non-zero exit status 127". 
The current .yml file needs update.  Installing gensim before 4.0 will be needed in your mallet enviornment (i.e., pip install gensim==3.8.3).  Pandas, numpy, scipy may be needed as well.*

View the top words in each topic with
```console
$ cut -f 1-10 -d " " results/mallet-speeches/topics.txt
```
