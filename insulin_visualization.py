#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
insulin_visualization.py - Insulin Expression Stability Analysis Module

This module performs statistical analysis and visualization of insulin expression
stability over time across different experimental conditions. The analysis includes
linear regression of log-transformed expression data to calculate decay rates and
statistical significance of temporal trends.

Key methods:
    - Log-linear regression to compute expression decay constants.
    - Ordinary Least Squares (OLS) regression for statistical modeling.
    - Comparative visualization of expression stability across conditions.
    - Statistical significance testing of decay rate differences.

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Jørgensen K, et al. (2018). Metabolic engineering of yeast for production of fuels and chemicals.
        FEMS Yeast Research, 18(7).
    [2] Nielsen J. (2019). Yeast systems biology: Model organism and cell factory.
        Biotechnology Journal, 14(9):1800421.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels.api as sm
import matplotlib as mpl

# Set plot sizes for publication-quality figures - DO NOT CHANGE (preserves output formatting)
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)         # Default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)      # Axes title fontsize
plt.rc('axes', labelsize=MEDIUM_SIZE)     # Axes labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # Tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # Tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # Legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # Figure title fontsize

mpl.rcParams['lines.markersize'] = 12


def load_data(file_path):
    """
    Load insulin expression data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing time-series expression data.

    Returns
    -------
    pandas.DataFrame
        Raw insulin expression data.
        
    Notes
    -----
    Expected CSV format includes columns for experiment type, time (in days),
    and expression levels (in mg/L).
    """
    return pd.read_csv(file_path)


def process_data(raw_data):
    """
    Process insulin expression data through normalization and log transformation.

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Raw time-series insulin expression data.

    Returns
    -------
    pandas.DataFrame
        Processed data with normalized and log-transformed expression values.
        
    Notes
    -----
    - Normalizes expression levels relative to 28.95 mg/L.
    - Filters out control experiment (SUC2) as in the original analysis.
    - Applies log transformation to enable linear regression analysis.
    """
    # CRITICAL: Preserve normalization factor to ensure output consistency
    raw_data.loc[:, 'expression_levels'] = raw_data.expression_levels / 28.95
    # CRITICAL: Exclude control experiment (SUC2)
    processed = raw_data[raw_data.experiment != 'SUC2'].copy()
    processed.loc[:, 'Expression Levels [log scale]'] = np.log(processed.expression_levels)
    processed.loc[:, 'Time [Days]'] = processed["Time(Days)"]
    return processed


def plot_data(processed_data, output_path):
    """
    Generate regression plots of insulin expression decay and calculate decay rates.

    Parameters
    ----------
    processed_data : pandas.DataFrame
        Processed insulin expression data.
    output_path : str
        Path where the output figure will be saved.

    Returns
    -------
    None
        The plot is saved to output_path and results are printed to console.

    Notes
    -----
    Applies log-linear regression, corresponding to an exponential decay model:
    y = y0 * exp(k*t) → ln(y) = ln(y0) + k*t, which allows linear regression analysis.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['P', 'X', 'o', 'v']
    decay_results = {}

    # Analyze experiments separately
    for ii, exp in enumerate(['ORIGINAL_SEQ', 'ESO', 'ARC15', 'CAF20']):
        df_curr = processed_data[processed_data.experiment == exp].copy()

        # Initial regression using scipy's linregress
        slope, intercept, r, p, sterr = scipy.stats.linregress(
            df_curr["Time [Days]"],
            df_curr["Expression Levels [log scale]"]
        )

        # Rename experiments for display
        display_name = exp
        if exp == 'ORIGINAL_SEQ':
            display_name = 'Original Sequence'
        elif exp == 'ESO':
            display_name = 'Sequence Optimization'

        # Create regression plot with seaborn
        sns.regplot(
            data=df_curr,
            x="Time [Days]",
            y="Expression Levels [log scale]",
            fit_reg=True,
            ax=ax,
            label=f'{display_name}, \ny={int(np.exp(intercept))}*exp({round(slope, 3)}T), R2={round(r ** 2, 2)}',
            marker=markers[ii]
        )

        # Rigorous regression using statsmodels OLS
        X = sm.add_constant(df_curr['Time [Days]'])
        model = sm.OLS(df_curr["Expression Levels [log scale]"], X).fit()

        decay_results[display_name] = {
            'slope': model.params[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared
        }
        print(f"\nExperiment {display_name}:")
        print(f"Slope (decay rate): {decay_results[display_name]['slope']:.4f}")
        print(f"P-value: {decay_results[display_name]['p_value']:.8f}")
        print(f"R-squared: {decay_results[display_name]['r_squared']:.4f}")

    ax.set_xlabel('Days in experiment')
    ax.set_ylabel('Log(Insulin titer in mg/L)')
    ax.legend()
    fig.savefig(output_path, dpi=300)


def main():
    """
    Main execution function for insulin expression stability analysis.

    Coordinates data loading, processing, analysis, and visualization.
    """
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, 'raw_data', 'RAW_DATA_insulin.csv')
    raw_data = load_data(data_path)
    processed = process_data(raw_data)
    output_path = os.path.join(script_dir, 'figures', 'insulin_results.png')
    plot_data(processed, output_path)
    print(f"Analysis complete. Figure saved to {output_path}")


if __name__ == "__main__":
    main()