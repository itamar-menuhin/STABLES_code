#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_gene_raw_data.py - Analysis of Gene Expression Fluorescence Measurements

This module analyzes fluorescence data from a 10-gene experiment by performing:
    • Normalization of fluorescence data relative to wild-type controls
    • Calculation of final/initial fluorescence ratios per experimental condition
    • Statistical comparisons using t-tests and Kruskal-Wallis tests
    • Visualization and saving of results

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Alon U. An Introduction to Systems Biology... (2019)
    [2] Kruskal WH, Wallis WA. JASA, 1952.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import warnings

# Set plot sizes for publication-quality figures
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

sns.set_theme(rc={"figure.dpi": 600})
warnings.filterwarnings("ignore")


def calculate_pvalue_between_experiments(df, exp1, exp2):
    """
    Calculate a one-tailed p-value for the difference in mean ratios between two experiments.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Contains columns 'experiment', 'mean_ratio', and 'std_error'.
    exp1 : str
        First experiment name.
    exp2 : str
        Second experiment name.
    
    Returns
    -------
    float
        One-tailed p-value.
    """
    ratio_exp1 = df[df.experiment == exp1].iloc[0, 1]
    error_exp1 = df[df.experiment == exp1].iloc[0, 2]
    ratio_exp2 = df[df.experiment == exp2].iloc[0, 1]
    error_exp2 = df[df.experiment == exp2].iloc[0, 2]
    tot_err = ((error_exp1 ** 2) + (error_exp2 ** 2)) ** 0.5
    t_stat = np.abs(ratio_exp1 - ratio_exp2) / tot_err
    p_value = 1 - stats.t.cdf(t_stat, 1)
    return p_value


def compute_mean_ratios_and_ttest(df, output_path):
    """
    Compute mean fluorescence ratios (final/initial) and perform t-tests between conditions.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Contains 'experiment', 'time', 'sample', and 'normalized_fluorescence'.
    output_path : str
        Directory where the graph and CSV will be saved.
    
    Returns
    -------
    None
    """
    experiments = df['experiment'].drop_duplicates()
    results = []

    # Calculate ratios for each experiment
    for experiment in experiments:
        curr_data = df[df['experiment'] == experiment][['time', 'sample', 'normalized_fluorescence']].dropna()
        initial_values = curr_data[curr_data['time'] == 1]['normalized_fluorescence'].values
        final_values = curr_data[curr_data['time'] == 15]['normalized_fluorescence'].values
        ratios = final_values / initial_values
        mean_ratio = np.mean(ratios)
        std_error = np.std(ratios) / np.sqrt(len(ratios))
        results.append({'experiment': experiment, 'mean_ratio': mean_ratio, 'std_error': std_error})

    results_df = pd.DataFrame(results).sort_values('mean_ratio', ascending=True)

    # Compare GFP with all other experiments
    ratio_GFP = results_df[results_df.experiment == 'GFP'].iloc[0, 1]
    error_GFP = results_df[results_df.experiment == 'GFP'].iloc[0, 2]
    ratio_coupled = results_df[results_df.experiment != 'GFP'].mean_ratio.mean()
    var_coupled = (results_df[results_df.experiment != 'GFP'].std_error ** 2).sum() / 100
    err_coupled = var_coupled ** 0.5
    tot_err = ((err_coupled ** 2) + (error_GFP ** 2)) ** 0.5
    t_stat = (ratio_coupled - ratio_GFP) / tot_err
    p_value = stats.t.cdf(t_stat, 9)

    print(f"T-statistic: {t_stat}")
    print(f"P-value (1-tailed): {1 - p_value}")

    # Generate p-values for each gene compared to CDC9
    for gene in list(results_df.experiment):
        print(f"P-value for CDC9 vs {gene}: {calculate_pvalue_between_experiments(results_df, 'CDC9', gene)}")

    # Plot bar graph with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(results_df['experiment'], results_df['mean_ratio'], yerr=results_df['std_error'],
            capsize=5, color='skyblue')
    plt.xlabel('Experiment')
    plt.ylabel('Mean Ratio (Final / Initial)')
    plt.title('Mean Ratio for 10-Gene Experiment')
    plt.xticks(rotation=45)
    plt.ylim([0.0, 1.2])
    plt.tight_layout()
    fig.savefig(output_path + '.png')

    # Save results to CSV
    results_df.sort_values('mean_ratio').to_csv(output_path + '.csv')


def perform_kruskal_wallis_test(data, time_threshold):
    """
    Perform the Kruskal-Wallis H-test on normalized fluorescence data for time points beyond a threshold.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Contains 'time', 'sample', 'experiment', and 'normalized_fluorescence'.
    time_threshold : int or float
        Only consider data with time >= time_threshold.
    
    Returns
    -------
    None
    """
    late_data = data[data['time'] >= time_threshold].copy()
    late_data['intervention'] = late_data.experiment
    late_data['value'] = late_data.normalized_fluorescence

    intervention_groups = [group['value'].values for name, group in late_data.groupby('intervention')]
    _, p_value_between = stats.kruskal(*intervention_groups)
    print(f"P-value for Kruskal-Wallis H-test: {p_value_between}")


def normalize_fluorescence_data(df):
    """
    Normalize fluorescence data relative to wild-type controls.
    
    This is done by subtracting the mean wild-type (WT) fluorescence at each time point
    (excluding sample 3) and then dividing by the initial fluorescence (time = 1).
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must include 'time', 'sample', 'experiment', and 'fluorescence'.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional column 'normalized_fluorescence'.
    """
    df_WT = df[df.experiment == 'WT']
    df_WT = df_WT[df_WT['sample'] != 3]
    df_WT = df_WT.groupby('time', as_index=False).fluorescence.mean().rename(columns={'fluorescence': 'WT'})

    df_exps = df[df.experiment != 'WT'].merge(df_WT, on='time')
    df_exps.loc[:, 'fluorescence'] = df_exps.fluorescence - df_exps.WT
    normalization_values = df_exps[df_exps['time'] == 1].set_index(['experiment', 'sample'])['fluorescence']

    def normalize(group):
        group['normalized_fluorescence'] = group.apply(
            lambda row: row['fluorescence'] / normalization_values.loc[(row['experiment'], row['sample'])],
            axis=1
        )
        return group

    normalized_df = df_exps.groupby(['experiment', 'sample'], group_keys=False).apply(normalize)
    return normalized_df


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    norm_fluor_path = os.path.join(script_dir, 'raw_data', 'normalized_fluorescence.csv')
    df_fluor = pd.read_csv(norm_fluor_path)

    # Reshape and normalize data
    df_fluor = pd.melt(df_fluor, id_vars=['time', 'sample'],
                       var_name='experiment', value_name='fluorescence')
    df_fluor = normalize_fluorescence_data(df_fluor)

    output_path = os.path.join(script_dir, 'processed_data', 'ratios')
    compute_mean_ratios_and_ttest(df_fluor, output_path)
    perform_kruskal_wallis_test(df_fluor, 15)
