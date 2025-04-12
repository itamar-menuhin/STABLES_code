#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fluorescence_analysis_2024.py - Fluorescent Protein Correlation Analysis Module

This module analyzes the correlation between GFP and mCherry fluorescence measurements
to evaluate consistency in dual-reporter systems in yeast. It performs:
    - Data loading and preprocessing
    - Correlation and regression analysis between fluorescence signals
    - Joint distribution visualization with marginal histograms on a log scale

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Shaner NC, et al. Nat Biotechnol. 2004.
    [2] Kremers GJ, et al. Int Rev Mol Cell Biol. 2021;373:187-233.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy

def load_data(file_path):
    """
    Load fluorescence measurement data from a CSV file.
    
    Parameters
    ----------
    file_path : str
        CSV file path containing fluorescence data.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with fluorescence measurements.
    """
    return pd.read_csv(file_path)

def plot_data(df_ground_truth, output_path):
    """
    Generate and save a joint plot for GFP vs. mCherry fluorescence.
    
    Parameters
    ----------
    df_ground_truth : pandas.DataFrame
        DataFrame with fluorescence measurements.
    output_path : str
        File path to save the output figure.
    """
    # Rename columns for better visualization and round values for clarity
    df_ground_truth = df_ground_truth.rename(columns={
        'GFP_fluorescence': 'GFP fluorescence',
        'RFP_fluorescence': 'mCherry fluorescence'
    }).round(2)

    # Use a consistent theme for publication-quality figures
    sns.set_theme(style="darkgrid")
    fig = sns.jointplot(
        x="GFP fluorescence", 
        y="mCherry fluorescence", 
        data=df_ground_truth, 
        kind='reg', 
        scatter_kws={'alpha': 0.1}, 
        marginal_kws={'log_scale': True}
    )
    
    # Calculate regression statistics using scipy
    slope, intercept, r, p, sterr = scipy.stats.linregress(
        df_ground_truth['GFP fluorescence'], 
        df_ground_truth['mCherry fluorescence']
    )
    print(f"Linear regression p-value: {p}")
    
    # Add regression information to the plot
    plt.text(
        10, 1000, 
        f'y = {round(intercept, 3)} + {round(slope, 3)}x\nR² = {round(r ** 2, 3)}'
    )
    
    # Maintain log scale for both axes to capture wide fluorescence ranges
    fig.ax_joint.set_ylim(ymin=1)
    fig.ax_joint.set_xscale('log')
    fig.ax_joint.set_yscale('log')
    
    # Save the resulting figure with high resolution
    fig.savefig(output_path, dpi=300)

def main():
    """
    Main execution function that orchestrates data loading and analysis.
    
    Loads fluorescence data and generates a correlation plot. Uses relative paths
    based on the script location for portability.
    """
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, 'raw_data', 'cleaned_data.csv')
    df_ground_truth = load_data(data_path)

    output_path = os.path.join(script_dir, 'figures', 'fluorescence_label_comparison.png')
    plot_data(df_ground_truth, output_path)
    print(f"Analysis complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()