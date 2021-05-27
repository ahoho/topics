
# coding: utf-8

# In[8]:

import json
import numpy as np
import pandas as pd

from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind


# In[3]:

data = json.load(open('topics/data/human/all_data/all_data.json', 'r'))


# In[4]:

def get_intrusion_overall_count_and_total_for_model(data, dataset, model):
    l = data[dataset][model]['metrics']['intrusion_scores_raw']
    counts, totals = [], []
    for x in l:
        counts.append(x.count(1))
        totals.append(len(x))
    return sum(counts), sum(totals)

intrusion_overall_counts_and_totals = {}
for dataset in data:
    intrusion_overall_counts_and_totals[dataset] = {}
    for model in data[dataset]:
        intrusion_overall_counts_and_totals[dataset][model] = (get_intrusion_overall_count_and_total_for_model(data, dataset, model))
#print(intrusion_overall_counts_and_totals)

def compute_proportions_test(counts_and_totals, dataset, model1, model2):
    d = counts_and_totals[dataset]
    stat, pval = proportions_ztest([d[model1][0], d[model2][0]],
                                   [d[model1][1], d[model2][1]],
                                  alternative='larger')
    return stat, pval
    


# In[5]:

def get_all_ratings_overall_for_model(data, dataset, model):
    l = data[dataset][model]['metrics']['ratings_scores_raw']
    return [rating for topic in l for rating in topic]

def compute_mannwhitney_two_models_ratings(data, dataset, model1, model2):
    l1 = get_all_ratings_overall_for_model(data, dataset, model1)
    l2 = get_all_ratings_overall_for_model(data, dataset, model2)
    stat, pval = mannwhitneyu(l1, l2, alternative='greater')
    return stat, pval


# In[6]:

def get_intrusion_count_and_total_for_conf_ann_model(data, dataset, model):
    l = data[dataset][model]['metrics']['intrusion_scores_raw']
    c = data[dataset][model]['metrics']['intrusion_confidences_raw']
    counts, totals = [], []
    for x, y in zip(l, c):
        t = 0
        count = 0
        for score, conf in zip(x, y):
            if conf:
                count += score
                t += 1
        counts.append(count)
        totals.append(t)
    return sum(counts), sum(totals)

intrusion_conf_counts_and_totals = {}
for dataset in data:
    intrusion_conf_counts_and_totals[dataset] = {}
    for model in data[dataset]:
        intrusion_conf_counts_and_totals[dataset][model] = (get_intrusion_count_and_total_for_conf_ann_model(data, dataset, model))
#print(intrusion_conf_counts_and_totals)

def compute_proportions_test(counts_and_totals, dataset, model1, model2):
    d = counts_and_totals[dataset]
    stat, pval = proportions_ztest([d[model1][0], d[model2][0]],
                                   [d[model1][1], d[model2][1]],
                                  alternative='larger')
    return stat, pval
    


# In[7]:

def get_all_ratings_conf_for_model(data, dataset, model):
    l = data[dataset][model]['metrics']['ratings_scores_raw']
    confs = data[dataset][model]['metrics']['ratings_confidences_raw']
    out = []
    for topic_scores, topic_confs in zip(l, confs):
        for rating, conf in zip(topic_scores, topic_confs):
            if conf:
                out.append(rating)
    return out

def compute_mannwhitney_two_models_ratings_conf(data, dataset, model1, model2):
    l1 = get_all_ratings_conf_for_model(data, dataset, model1)
    l2 = get_all_ratings_conf_for_model(data, dataset, model2)
    stat, pval = mannwhitneyu(l1, l2, alternative='greater')
    return stat, pval


# In[15]:

def get_all_npmi_scores_for_model(data, dataset, model):
    l = data[dataset][model]['metrics']['c_npmi_10_full']
    return l

def compute_ttest_two_models_ratings_npmi(data, dataset, model1, model2):
    l1 = get_all_npmi_scores_for_model(data, dataset, model1)
    l2 = get_all_npmi_scores_for_model(data, dataset, model2)
    stat, pval = ttest_ind(l1, l2, alternative='greater')
    return stat, pval


# ## For both intrusion and ratings task
# 
# ### Get 2 tables each - one for wikitext and one for nytimes
# 
# ### Each table is 3 by 3 pairwise comparisons of models (mallet, dvae, etm), read as: p-value for model 1 (row) being rated higher than model 2 (column), with value boldfaced for p < 0.05

# In[22]:

dataset_to_latex_call = {'wikitext': '\\abr{wiki}', 'nytimes': '\\abr{nyt}'}


# In[23]:

def get_val_result_rows(counts_and_totals, dataset, models, alpha, task, anns = 'all'):
    result_rows = ['', '', '']
    for i, model1 in enumerate(models):
        row = result_rows[i]
        row = row + '\\abr{' + model1 + '} & '
        for j, model2 in enumerate(models):
            if model1!=model2:
                if task=='intrusion':
                    stat, pval = compute_proportions_test(counts_and_totals, dataset, model1, model2)
                elif task=='ratings':
                    if anns == 'all':
                        stat, pval = compute_mannwhitney_two_models_ratings(data, dataset, model1, model2)
                    elif anns == 'confident_only':
                        stat, pval = compute_mannwhitney_two_models_ratings_conf(data, dataset, model1, model2)
                elif task=='npmi':
                    stat, pval = compute_ttest_two_models_ratings_npmi(data, dataset, model1, model2)
                #pval_str = format(pval, '.2e')
                if pval < 0.001:
                    pval_str = "$<$ 0.001"
                else:
                    pval_str = str(round(pval, 3))
                if pval < alpha:
                    row = row + '\\textbf{' + pval_str + '}'
                else:
                    row = row + pval_str
            else:
                row = row + ' '
            if j < len(models) - 1:
                row = row + ' & '
            else:
                row = row + ' \\\ '
        result_rows[i] = row
    return result_rows


# In[24]:

def create_latex_table(counts_and_totals, dataset, models, alpha, task, anns = 'all'):
    rows = ["\\begin{table}[h]",
            "\centering",
            "\\begin{tabular}{lrrr}",
            "\\toprule"]
    headings = ' & '.join([' '] + ['\\abr{' + m + '}' for m in models]) + ' \\\ '
    rows.append(headings)
    rows.append("\midrule")
    
    rows = rows + get_val_result_rows(counts_and_totals, dataset, models, alpha, task, anns)
    rows.append("\\bottomrule")
    rows.append("\end{tabular}")
    if task=='npmi':
        caption = "\caption{Pairwise Model comparison based on Automated Metric of \\abr{npmi}. For each cell, value shown is the p-value for one-sided t-test for the hypothesis: Model in the row scores significantly higher than Model in the column. Boldfaced values show significant difference at $p < 0.05$."
        caption = caption + ' Dataset used is ' + '\\textbf{' + dataset_to_latex_call[dataset] + '}.}'
    else:
        caption = "\caption{Pairwise Model comparison based on Human Evaluation via the \\textbf{" + task.title() + '} task. For each cell, value shown is the p-value for one-sided '
        if task=='intrusion':
            caption = caption + ' Proportions test for the hypothesis: Model in the row scores significantly higher than Model in the column. Boldfaced values show significant difference at $p < 0.05$.'
        elif task=='ratings':
            caption = caption + ' Mann-Whitney U test for the hypothesis: Model in the row scores significantly higher than Model in the column. Boldfaced values show significant difference at $p < 0.05$.'
        caption = caption + ' Dataset used is ' + '\\textbf{' + dataset_to_latex_call[dataset] + '}.}'
        if anns == 'confident_only':
            caption = caption[:-1] + " \\textbf{Human scores considered ONLY where annotator was familiar with the terms appearing in the topic.}}"
    rows.append(caption)
    rows.append("\label{tab:" + dataset + '_' + task + "_pairwise_model_comparison}")
    rows.append("\end{table}")
    return rows


# In[25]:

alpha = 0.05
#datasets = ['wikitext', 'nytimes']
models = ['mallet', 'dvae', 'etm']


# In[26]:

print('\n'.join(create_latex_table(intrusion_overall_counts_and_totals,
                                   'wikitext', models, alpha, 'intrusion')))


# In[27]:

print('\n'.join(create_latex_table(intrusion_overall_counts_and_totals,
                                   'wikitext', models, alpha, 'ratings')))


# In[28]:

print('\n'.join(create_latex_table(intrusion_overall_counts_and_totals,
                                   'nytimes', models, alpha, 'intrusion')))


# In[29]:

print('\n'.join(create_latex_table(intrusion_overall_counts_and_totals,
                                   'nytimes', models, alpha, 'ratings')))


# In[ ]:




# In[30]:

print('\n'.join(create_latex_table(intrusion_conf_counts_and_totals,
                                   'wikitext', models, alpha, 'intrusion', 'confident_only')))


# In[31]:

print('\n'.join(create_latex_table(intrusion_conf_counts_and_totals,
                                   'wikitext', models, alpha, 'ratings', 'confident_only')))


# In[32]:

print('\n'.join(create_latex_table(intrusion_conf_counts_and_totals,
                                   'nytimes', models, alpha, 'intrusion', 'confident_only')))


# In[33]:

print('\n'.join(create_latex_table(intrusion_conf_counts_and_totals,
                                   'nytimes', models, alpha, 'ratings', 'confident_only')))


# In[ ]:




# In[34]:

print('\n'.join(create_latex_table(intrusion_overall_counts_and_totals,
                                   'wikitext', models, alpha, 'npmi')))


# In[35]:

print('\n'.join(create_latex_table(intrusion_overall_counts_and_totals,
                                   'nytimes', models, alpha, 'npmi')))


# In[ ]:



