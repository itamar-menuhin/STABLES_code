#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xgb_feature_importance.py - XGBoost SHAP Value Visualization Module

This module analyzes and visualizes feature importance from XGBoost models using
SHAP (SHapley Additive exPlanations) values. It loads pre-computed SHAP values,
calculates mean feature contributions, and generates publication-ready bar plots.

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions.
        Advances in Neural Information Processing Systems. 2017;30:4765-4774.
    [2] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System.
        KDD '16: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016:785-794.
"""

import os
from os import path
import pandas as pd
import seaborn as sns
import warnings

# Set visualization parameters for publication-quality figures
sns.set_theme(rc={"figure.dpi": 600}, style="whitegrid")
warnings.filterwarnings("ignore")


def main(shapley_path, output_path):
    """
    Calculate and visualize feature importance using SHAP values from XGBoost models.
    
    Parameters
    ----------
    shapley_path : str
        Directory containing the pre-calculated SHAP values.
    output_path : str
        Directory where the output figure will be saved.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # CRITICAL: Load SHAP values with consistent column structure
    df_shapley = pd.read_csv(path.join(shapley_path, 'XGB_shapley_mean.csv'), index_col=0)
    df_shapley['shapley_mean'] = df_shapley.mean(axis=1)
    
    # Select top 20 features by mean SHAP value and round results
    df_shapley = (
        df_shapley.sort_values('shapley_mean', ascending=False)
                   .head(20)
                   .round(2)
                   .reset_index(names='selected_feature')
    )
    
    # Reshape data for visualization
    df_shapley_melt = df_shapley.melt(
        id_vars='selected_feature', 
        value_vars=[str(x) for x in range(1, 21)],
        var_name='split', 
        value_name='Shap_value'
    )
    
    # Generate a horizontal bar plot with error bars (standard deviation)
    g = sns.catplot(
        data=df_shapley_melt, kind="bar", orient='h',
        y="selected_feature", x="Shap_value",
        errorbar="sd", palette="dark", alpha=0.6, height=6, aspect=2
    )
    g.despine(left=True)
    g.set_axis_labels("Shapley Value", "Selected Features")
    g.set_titles("XGBoost Feature Importance")
    
    # Save the figure with high resolution
    out_file = path.join(output_path, 'XGB_shapley_bar_abs.png')
    g.figure.savefig(out_file)
    print(f"Feature importance visualization saved to {out_file}")


if __name__ == "__main__":
    # Set paths relative to this script for reproducibility
    script_dir = os.path.dirname(__file__)
    shapley_path = path.join(script_dir, 'raw_data')
    output_path = path.join(script_dir, 'figures')
    main(shapley_path, output_path)