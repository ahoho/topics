from tqdm import tqdm

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

def gen_measure_name(coherence_measure, window_size, top_n):
    """
    Make a unique measure name from the arguments
    """
    measure_name = f"{coherence_measure}_win{window_size}_top{top_n}"
    return measure_name


def calculate_coherence(
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

    measure_name = gen_measure_name(coherence_measure, cm.window_size, top_n)
    return measure_name, float(mean), [float(i) for i in confirmed_measures]