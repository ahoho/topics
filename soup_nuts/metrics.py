import random
from multiprocessing import Pool
from collections import defaultdict
from itertools import combinations

import rbo
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist, jensenshannon
from scipy.optimize import linear_sum_assignment
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

from sklearn.metrics import rand_score, normalized_mutual_info_score , adjusted_rand_score


def _gen_measure_name(coherence_measure, window_size, top_n):
    """
    Make a unique measure name from the arguments
    """
    measure_name = f"{coherence_measure}_win{window_size}_top{top_n}"
    return measure_name


def _summarize(data):
    return {**pd.Series(data).describe().to_dict(), "sum": np.sum(data)}


def coherence(
    topics,
    vocab,
    reference_text,
    coherence_measure,
    window_size,
    top_n,
):
    """
    Calculates coherence for a single model
    """
    data_dict = Dictionary([vocab])
    topics = [t[:top_n] for t in topics]
    
    cm = CoherenceModel(
        topics=topics,
        texts=tqdm(reference_text),
        dictionary=data_dict,
        coherence=coherence_measure,
        window_size=window_size,
    )

    confirmed_measures = cm.get_coherence_per_topic()
    mean = cm.aggregate_measures(confirmed_measures)

    measure_name = _gen_measure_name(coherence_measure, cm.window_size, top_n)
    return measure_name, float(mean), [float(i) for i in confirmed_measures]


def purity(model_labels, gold_labels): 
    """
    Calculates the Purity metric as described in https://aclanthology.org/P16-1110/
    "ALTO: Active Learning with Topic Overviews for Speeding Label Induction and Document Labeling"
    
    For sanity check - The purity of any two user labels should be 1
    """
    assert len(model_labels) == len(gold_labels)

    # somewhat faster than a pure-python implementation
    purity_sum = (
        pd.DataFrame({"pred": model_labels, "true": gold_labels})
          .groupby(["pred", "true"], as_index=False)
          .size()
          .groupby("pred")["size"]
          .max()
          .sum()
    )
    return purity_sum / len(model_labels)


def rbo_dist(x, y, pval):
    """Rank-biased overlap distance. Assumes x, y truncated to top n items"""
    return 1 - rbo.RankingSimilarity(x, y).rbo(p=pval)

def avg_jcrd_agreement(x, y):
    """Average Jaccard distance. Assumes x, y truncated to top n items"""
    score = 0.0
    max_set_size = len(x)
    for depth in range(1, max_set_size + 1):
        s1 = set(x[:depth])
        s2 = set(y[:depth])
        jcrd = len(s1 & s2)/len(s1 | s2)
        score += jcrd
    score = score/max_set_size
    return 1.0 - score


def unique_doc_words_over_runs(doc_topic_runs, topic_word_runs, top_n=15, hard_assignment=True, summarize=False):
    """
    Given a collection of estimates of document-topic distributions over a set of runs,
    determine how stable the topic assignments are per document by taking the union of
    the set of top words predicted for each document

    To determine what words are predicted for a given document, we use the reconstructed
    bag-of-words: the admixture of (global) topic-word probabilities, weighted by the 
    document's topic probabilities

    Setting `hard_assignment` uses the top words from the most-probable topic for the
    document
    """
    runs = len(doc_topic_runs)
    n = doc_topic_runs[0].shape[0]
    
    top_words_over_runs = np.zeros((n, runs * top_n))
    # for each run, determine the set of words that are predicted for each topic
    for i, (doc_topic, topic_word) in enumerate(zip(doc_topic_runs, topic_word_runs)):
        if hard_assignment:
            # find the top words per topic, then index by documents' top topic
            top_words = (-topic_word).argsort(1)[doc_topic.argmax(1), :top_n]
        else:
            # calculate the mixture over topic-word probabilities per doc
            top_words = (-(doc_topic @ topic_word)).argsort(1)[:, :top_n]
        # store them in an array
        top_words_over_runs[:, i*top_n:(i+1)*top_n] = top_words

    # then, determine the unique number of words predicted for each document
    # stackoverflow.com/questions/48473056/number-of-unique-elements-per-row-in-a-numpy-array
    nunique = np.count_nonzero(np.diff(np.sort(top_words_over_runs, axis=1)), axis=1) + 1

    # finally, normalize between the lowest possible and highest possible number of unique terms
    punique = (nunique - top_n) / (top_n * (runs - 1))
    if summarize: # summary is _over n_ (not runs)
        return _summarize(punique)

    return nunique, punique


def unique_topic_words_over_runs(topic_word_runs, top_n=15, summarize=False):
    """
    Given a collection of estimates of topic-word distributions, calculate
    how stable the topics are by comparing the set of top words
    """
    runs = len(topic_word_runs)
    k = topic_word_runs[0].shape[0]
    max_count_digits = int(10 ** np.ceil(np.log10(k)))

    unique_words = set()
    for topic_word in topic_word_runs:
        top_words = (-topic_word).argsort(1)[:, :top_n].reshape(-1)
        word_counter = defaultdict(lambda: -1)
        # problem: if repeated words appear across the topics in a single run, 
        # this will underestimate the number of unique words produced over runs.
        # to solve: a word can be repeated at most `k` times in a run (once per topic).
        # count each occurrence per run as a unique  term, i.e., "mouse_0", "mouse_1", etc.
        # this count gets stored in the first `max_count_digits`, so if "mouse" has index
        # 155, the 3rd appearance is coded as 15502.
        for w in top_words:
            word_counter[w] += 1
            w_c = w*max_count_digits + word_counter[w]
            unique_words.add(w_c)

    nunique = len(unique_words)
    words_per_run = k * top_n
    # normalize the score between lowest and highest possible number of unique terms
    punique = (nunique - words_per_run) / (words_per_run * (runs - 1))
    if summarize: # `punique` is a single value, but this unifies the API
        return _summarize(punique)

    return nunique, punique


def topic_dists_over_runs(
    *, # enforce named arguments to avoid ambiguity
    doc_topic_runs=None,
    topic_word_runs=None,
    metric="jensenshannon",
    sample_n=1.0,
    summarize=False,
    seed=None,
    workers=1,
    top_n_items=None, # for rank based metrics ("rbo", "jaccard"), select top n items
    tqdm_kwargs={},
):
    """
    Estimate the stability of topics by calculating the distance between topics across runs.

    Works on either topic-word or document-topic estimates, where "topics" are considered
    the vector for each topic dimension in the estimate. That is, for a topic-word estimate
    the vector is the size of the vocabulary, |V|, and for a doc-topic estimate it's the number 
    of documents N.

    For each of the (runs*(runs-1))/2 pairs of runs, there is a run_a and a run_b with
    associated estimates est_a and est_b.

    We take the pairwise distances between the k topic vectors contained in est_a and est_b,
    finding the minimum weight match between the topic pairs.

    To speed up computation, can set `sample_n` to use only a subset of possible combinations.

    TODO:
        - does pairwise js-distance depend on whether betas came from a softmax vs. some
    other method (e.g., gibbs?). does this matter?
        - does it make sense to have a pairwise spearman?
    """
    if topic_word_runs is not None and doc_topic_runs is not None:
        raise ValueError("Supply either `topic_word_runs` or `doc_topic_runs`, not both")
    
    # prepare the estimates
    estimates = doc_topic_runs if doc_topic_runs is not None else topic_word_runs
    for i in range(len(estimates)):
        x = estimates[i]
        # apprently float64 faster for cdist,stackoverflow.com/a/50671733/5712749
        x = x.astype(np.float64)
        if doc_topic_runs is not None:
            x = x.T
        if metric == "jensenshannon" and not np.allclose(x.sum(1), 1):
            x = x/x.sum(1, keepdims=True)

        if top_n_items == "auto":
            top_n_items = x.shape[1] // x.shape[0]
        
        if top_n_items is not None: # for rank-based metrics
            x = (-x).argsort(1)[:, :top_n_items]
        estimates[i] = x
    
    # sample the combinations of runs
    runs = len(estimates)
    combins = list(combinations(range(runs), 2))
    sample_n = sample_n if sample_n > 1 else int(sample_n * len(combins))
    random.seed(seed)
    random.shuffle(combins)
    combins = combins[:sample_n]

    # compute distances    
    # for each run pair, find the minimum global distances
    if workers <= 1:
        # first, initialize the matrix in which to store the distances
        num_topics = estimates[0].shape[0]
        min_dists = np.zeros((len(combins), num_topics))
        for i, (idx_a, idx_b) in enumerate(tqdm(combins, **tqdm_kwargs)):
            args = (estimates[idx_a], estimates[idx_b], metric)
            min_dists[i] = _min_total_topic_dist(args)
    else:
        with Pool(processes=workers) as pool:
            args = [(estimates[idx_a], estimates[idx_b], metric) for idx_a, idx_b in combins]
            result = pool.imap_unordered(_min_total_topic_dist, args)
            min_dists = np.array([r for r in tqdm(result, total=sample_n, **tqdm_kwargs)])

    min_dists = np.sort(min_dists, axis=1)
    if summarize:
        # not totally obvious how to report a summary
        # for now: we take the total cost and report summary over runs
        # could be an issue if different models' distrubtions have different entropies
        return _summarize(min_dists.sum(1))

    return min_dists


def _min_total_topic_dist(args):
    """Helper function to find the minimum total cost TODO: quicker approximation?"""
    x, y, metric = args
    dists = cdist(x, y, metric=metric) # get distances: produces a [k x k] "cost" matrix
    row_idx, col_idx = linear_sum_assignment(dists) # minimize the global match cost
    return dists[row_idx, col_idx]


def doc_words_dists_over_runs(
    doc_topic_runs,
    topic_word_runs,
    batchsize=1000,
    sample_n=1,
    workers=1,
    seed=None,
    tqdm_kwargs={}
):
    """
    For each document, calculate the jensen-shannon distance between its predicted word
    probabilities (i.e., reconstructed BoW) in each run

    TODO: since this is not a true forward pass of the model (e.g., in some models,
    `topic_word` is not normalized; a softmax is applied after `doc_topic @ topic_word`).
    Hence, this may introduce some problems---worth revisiting.
    """
    runs = len(doc_topic_runs)
    n = doc_topic_runs[0].shape[0]
    v = topic_word_runs[0].shape[1]

    assert(np.allclose(topic_word.sum(1).max(), 1)) # should be normalized

    # create the document-word estimates
    doc_word_probs = np.zeros((runs, n, v))
    for run_i, (doc_topic, topic_word) in enumerate(zip(doc_topic_runs, topic_word_runs)):
        for j in range(n // batchsize + 1):
            bs = np.s_[j*batchsize:(j+1)*batchsize]
            # NB: assumes topic-word is normalized, which will violate the 
            # true model for d-vae, scholar, and others
            p_hat = doc_topic[bs] @ topic_word
            doc_word_probs[run_i][bs] = p_hat

    # sample the combinations of runs
    combins = list(combinations(range(runs), 2))
    sample_n = sample_n if sample_n > 1 else int(sample_n * len(combins))
    random.seed(seed)
    random.shuffle(combins)
    combins = combins[:sample_n]
    
    doc_word_dists = np.zeros((len(combins), n))
    for i, (idx_a, idx_b) in enumerate(tqdm(combins, **tqdm_kwargs)):
        # may need to dispatch to workers; axis arg requires scipy 1.7.
        dists = jensenshannon(doc_word_probs[idx_a], doc_word_probs[idx_b], axis=1)
        doc_word_dists[i] = dists

    return doc_word_dists
