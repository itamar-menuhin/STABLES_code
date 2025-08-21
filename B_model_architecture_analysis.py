#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap_analysis_20seeds.py - Bootstrapping analysis for prediction model evaluation

This module performs bootstrapping analysis to compute model performance metrics,
visualize quantile frequencies in heatmaps, and produce summary boxplots.
It aggregates results for further evaluation.

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]
"""

import os
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Set up figure and font parameters for consistency
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Set working directory and file path for predictions
path_all_predictions = os.path.join(os.path.dirname(__file__), 'data', 'processed_data', 'all_predictions_20seeds_lean.csv')


def label_top(df, col, threshold, mode='percent'):
    """
    Label top entries in df for a given column.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to label.
    col : str
        Column used for ranking.
    threshold : int or float
        If mode=='percent', threshold is percentage; otherwise it is an absolute count.
    mode : {'percent', 'count'}, default 'percent'
        Determines threshold interpretation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with new column "{col}_top" set to 1 for top entries.
    """
    df_sorted = df.sort_values(by=col, ascending=False).reset_index(drop=True)
    count = int(len(df_sorted) * threshold / 100) if mode == 'percent' else threshold
    df_sorted[f'{col}_top'] = (df_sorted.index < count).astype(int)
    return df_sorted


def roc_auc_with_label(df_pred_label, n=5):
    """
    Compute ROC AUC based on top n% predicted and actual values.
    
    Parameters
    ----------
    df_pred_label : pandas.DataFrame
        DataFrame with columns 'preds' and 'label'.
    n : int, default 5
        Percentage threshold for top labels.
    
    Returns
    -------
    float
        ROC AUC score (binary labels derived from top-n).
    """
    temp_df = label_top(df_pred_label, 'preds', n, mode='percent')
    temp_df = label_top(temp_df, 'label', n, mode='percent')
    return metrics.roc_auc_score(temp_df['label_top'], temp_df['preds_top'])


def max_fluor_top3(df_pred_label):
    """
    Return the maximum label value among the top 3 'preds' entries.
    """
    return df_pred_label.nlargest(3, 'preds')['label'].max()


def quantile_fluor_top_i(df_pred_label, i=3):
    """
    Calculate quantile frequency from top i predicted values.
    
    Parameters
    ----------
    df_pred_label : pandas.DataFrame
        DataFrame with 'preds' and 'label'.
    i : int, default 3
        Top i entries to consider.
    
    Returns
    -------
    float
        Score computed from ranking of the top i labels.
    """
    total = df_pred_label.shape[0]
    top_preds = df_pred_label.nlargest(i, 'preds')
    sorted_label = df_pred_label.sort_values('label', ascending=False)\
        .reset_index(drop=True).reset_index(names='true_rank')
    merged = sorted_label.merge(top_preds, how='inner', on=list(df_pred_label.columns))
    min_rank = merged['true_rank'].min() if not merged.empty else total
    return 1 - (min_rank / total)


def accuracy_top_recommendations(df_pred_label, num_top_recommendations=3, percentile_to_consider_success=1):
    """
    Compute the recommendation accuracy based on top recommendations.
    """
    temp_df = label_top(df_pred_label, 'preds', num_top_recommendations, mode='count')
    temp_df = label_top(temp_df, 'label', percentile_to_consider_success, mode='percent')
    successes = temp_df[(temp_df['preds_top'] == 1) & (temp_df['label_top'] == 1)].shape[0]
    return successes / num_top_recommendations


def corr_results(df_pred_label, method='spearman'):
    """
    Compute correlation between predictions and labels.
    """
    return df_pred_label['preds'].corr(df_pred_label['label'], method=method)


def bootstrap_analysis_single_model(df_pred_label, group_name, sample_size):
    """
    Perform bootstrapping analysis for a single model.
    
    Parameters
    ----------
    df_pred_label : pandas.DataFrame
        Data for a single model.
    group_name : tuple
        Tuple (model_name, train_on).
    sample_size : int
        Size of each bootstrap sample.
    
    Returns
    -------
    pandas.DataFrame
        Bootstrap results with performance metrics.
    """
    num_iters = df_pred_label.shape[0]
    samples = []
    sample_size = sample_size or num_iters
    
    for ii in range(num_iters):
        if ii % 1000 == 0:
            print(ii)
        bootstrap_sample = df_pred_label.sample(sample_size, replace=True)
        sample_scores = (
            roc_auc_with_label(bootstrap_sample),
            max_fluor_top3(bootstrap_sample),
            quantile_fluor_top_i(bootstrap_sample, i=3),
            quantile_fluor_top_i(bootstrap_sample, i=1),
            corr_results(bootstrap_sample),
            corr_results(bootstrap_sample, method='pearson')
        )
        samples.append(sample_scores)
    
    df_boot = pd.DataFrame.from_records(
        samples,
        columns=['auc', 'max_top3', 'quantile_top3', 'quantile_top1', 'spearman_corr', 'pearson_corr']
    )
    df_boot['model'] = group_name[0]
    df_boot['train_on'] = group_name[1]
    return df_boot[['model', 'train_on', 'auc', 'max_top3', 'quantile_top3', 'quantile_top1', 'spearman_corr', 'pearson_corr']]


def bootstrap_all_models(df, sample_size=0, mode='mean'):
    """
    Perform bootstrapping analysis across all models.
    """
    df_new = df.copy()
    if mode == 'mean':
        df_new = df_new.groupby(['model', 'train_on', 'gene'], as_index=False)[['preds', 'label']].mean()
    
    bootstrap_list = []
    for group, group_df in df_new.groupby(['model', 'train_on']):
        boot_df = bootstrap_analysis_single_model(group_df, group, sample_size)
        print(boot_df)
        bootstrap_list.append(boot_df)
    
    return pd.concat(bootstrap_list, ignore_index=True)


def visualize_single_model(df, model='KNN_XGB', output_path='figures'):
    """
    Visualize histogram plots for a single model.
    """
    df_model = df[(df.model == model) & (df.train_on == 'all')]
    for col, binrange in zip(['quantile_top3', 'quantile_top1', 'spearman_corr'],
                             [(0.9, 1.0), (0.7, 1.0), (0.46, 0.54)]):
        fig, ax = plt.subplots()
        sns.histplot(data=df_model, x=col, ax=ax, binwidth=0.005, stat='percent', binrange=binrange)
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, f'single_model_{col}.png'))


def calculate_quantile_frequencies(df, model_col='model', quantile_col='quantile_top3'):
    """
    Calculate quantile frequency distributions for each model.
    """
    quantile_ranges = [(0.0, 0.25, '0.0-0.25'),
                       (0.25, 0.5, '0.25-0.5'),
                       (0.5, 0.75, '0.5-0.75'),
                       (0.75, 0.9, '0.75-0.9'),
                       (0.9, 0.95, '0.9-0.95'),
                       (0.95, 0.99, '0.95-0.99'),
                       (0.99, 1.01, '0.99-1.0')]
    results = {model: {label: ((group[quantile_col] >= low) & (group[quantile_col] < high)).mean()
                       for low, high, label in quantile_ranges}
               for model, group in df.groupby(model_col)}
    return pd.DataFrame.from_dict(results, orient='index')


def plot_quantile_distribution(freq_df, output_path, add_to_title=', max on top 3 predictions',
                               figsize=(12, 8), cmap='YlGnBu'):
    """
    Plot a heatmap of quantile frequency distributions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(freq_df, annot=True, cmap=cmap, fmt='.1%', cbar_kws={'label': 'Frequency'},
                linewidths=0.5, square=True, ax=ax)
    plt.title('Quantile Distribution Across Models' + add_to_title, fontsize=16, pad=20)
    plt.xlabel('Quantile Ranges', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Ensure output_path is in figures folder
    fig.savefig(os.path.join('figures', os.path.basename(output_path)))
    return


# ---- Main execution ----

df_all_preds = pd.read_csv(path_all_predictions, index_col=0).fillna('none')
df_all_preds = df_all_preds[df_all_preds.train_on == 'all']
df_bootstrap1 = bootstrap_all_models(df_all_preds, sample_size=6000)

# Rename models as required
dict_rename_model = {
    'ElasticNet': ['ElasticNet_KNN', 'ElasticNet_XGB'],
    'KNN': ['ElasticNet_KNN', 'KNN_XGB'],
    'XGB': ['ElasticNet_XGB', 'KNN_XGB']
}
renamed_dfs = []
for mod in dict_rename_model:
    df_curr = df_all_preds[df_all_preds.model == mod]
    df1 = df_curr.copy()
    df2 = df_curr.copy()
    df1.loc[:, 'model'] = dict_rename_model[mod][0]
    df2.loc[:, 'model'] = dict_rename_model[mod][1]
    renamed_dfs.extend([df1, df2])
df_all_preds2 = pd.concat(renamed_dfs)
df_bootstrap2 = bootstrap_all_models(df_all_preds2, sample_size=6000)

# Combine bootstrap results
df_bootstrap = pd.concat([df_bootstrap1, df_bootstrap2], ignore_index=True)
name_append = '_together'

# Generate boxplots for defined metrics
boxplots = [
    ('auc', None, ''),
    ('max_top3', None, '_all'),
    ('spearman_corr', None, '_all'),
    ('pearson_corr', None, '_all'),
    ('quantile_top3', (0.9, 1), '_all'),
    ('quantile_top1', (0.5, 1), '_all')
]
for col, ylim, extra in boxplots:
    fig, ax = plt.subplots()
    sns.boxplot(data=df_bootstrap[df_bootstrap.train_on == 'all'], x="model", y=col, ax=ax)
    fig.suptitle(col)
    plt.xticks(rotation=15)
    if ylim:
        plt.ylim(*ylim)
    fig.savefig(os.path.join('figures', f'{col}{name_append}{extra}.png'))

# Plot quantile distribution heatmaps
df_compare = df_bootstrap[df_bootstrap.train_on == 'all']
aa_top3 = calculate_quantile_frequencies(df_compare, quantile_col='quantile_top3')
plot_quantile_distribution(aa_top3, 'figures/top3_quantile_all.png', figsize=(12, 8), cmap='YlGnBu')
aa_top1 = calculate_quantile_frequencies(df_compare, quantile_col='quantile_top1')
plot_quantile_distribution(aa_top1, 'figures/top1_quantile_all.png', add_to_title=', top prediction', figsize=(12, 8), cmap='YlGnBu')

# Visualize single model results
visualize_single_model(df_bootstrap2)
