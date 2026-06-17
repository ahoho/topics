* Preprocess
* Run 
* Metrics
*

# An opinionated soup-to-nuts guide to topic modeling

This year marks the 20th anniversary of LDA. Topic modeling is still important [elaborate]

Why another guide? [Elaborate]      

New topic model variants will promise improvements thanks to BERT, Sentence Transformers, diffusion, or whatever. Before you use them, you should ask: _where's the human evaluation_? Until we have good proxies for topic coherence, take any claims of better NPMI with a big grain of salt.

For one thing, datasets, metrics, preprocessing decisions, and baselines are all unstandardized. For another, automated coherence metrics like NPMI aren't very predictive of human interpretability (the thing they're designed to correlate with). You can see more information in our paper.

So, what do we do? The guide & code aims to serve the needs of two different groups:
1. **Practitioners**: people who want a reasonable and accessible topic model that will produce reliable & interpretable results on their dataset of choice
2. **Methods researchers** who want stanardized baselines, datasets, and metrics so that they can fairly evaluate new methods

I saw that it is "opinionated" because two decades of research and tens of thousands of citations mean that the field has developed a set of best practices (even if they are sometimes ignored). That said, I do not know every part of the literature and am very open to suggestions and changes! [Link to Maria Antoniak]

The structure follows a pretty straightforward path: you install packages, preprocess data (I provide already-preprocessed benchmark data), 

Let's get started!

# Setup
## Prerequisites

You should be able to run everything on a standard laptop; everything should work on linux/Windows/macOS. I assume some knowledge of the command line.

## Installation

Everything is wrapped up into a package called `soup-nuts`.

First install [`uv`](https://docs.astral.sh/uv/) (e.g., `brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`).

Then, from the repository root, run

```console
$ uv sync
```

This creates an isolated `.venv/` on a uv-managed Python 3.12 and installs `soup-nuts` with all of its dependencies (pinned in `uv.lock`). uv manages the interpreter and virtualenv for you.

Check the installation with 

```console
$ uv run soup-nuts --help
```

`uv run` uses the project environment automatically (nothing to activate). If you have issues, you can also run via `uv run python -m soup_nuts.main <command name>`.

##