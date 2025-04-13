#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_apply.py - Model application for gene expression predictions

This module applies pre-trained models to processed feature data. It performs optional PCA
transformations on specific families of features and then predicts gene expression using
models such as XGBoost or alternatives.

Outputs:
  - Scaled data CSV
  - PCA-transformed data CSV (if applicable)
  - Predictions CSV (including mean predictions)

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]
"""

import os
from os import path
import pandas as pd
import pickle
import xgboost as xgb
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def is_family_amino_kmer(col_name, isTarget=False):
    """Return True if col_name is a valid 3-character amino k-mer.
    
    For target features, the 'target_' prefix is removed first.
    """
    if isTarget:
        if col_name.startswith('target_'):
            col_name = col_name.split('target_', 1)[1]
        else:
            return False
    return len(col_name) == 3 and col_name[1] == '_' and col_name[2] in ['3', '4', '5']


def is_family_by_start(col_name, start):
    """Return True if col_name starts with the specified prefix."""
    return col_name.startswith(start)


def family_sparser(df_data, family_name, family_cols, mode, pca):
    """Replace family features with their PCA projection."""
    if not family_cols or not mode:
        return df_data
    other_cols = [col for col in df_data.columns if col not in family_cols]
    df_irrelevant = df_data[other_cols]
    df_relevant = df_data[family_cols]
    n_components = len(family_cols) // 2 if mode == 'half' else (3 * len(family_cols)) // 4
    df_pca = pd.DataFrame(
        pca.transform(df_relevant),
        columns=[f"{family_name}_{i}" for i in range(n_components)],
        index=df_relevant.index
    )
    return pd.concat([df_irrelevant, df_pca], axis=1)


def generate_PCA_data(df_data, seed_directory, model):
    """Apply PCA transformation on selected feature families if required."""
    if model in ['KNN', 'SVM', 'linear']:
        return df_data
    mode = 'half' if model == 'ElasticNet' else 'sqrt'
    pca_path = path.join(seed_directory, 'PCA')
    features = list(df_data.columns)

    # Process amino_kmers
    amino_kmers = [col for col in features if is_family_amino_kmer(col)]
    with open(path.join(pca_path, f'amino_kmers_pca_{mode}.pkl'), 'rb') as handle:
        pca_amino = pickle.load(handle)
    df_data = family_sparser(df_data, 'amino_kmers', amino_kmers, mode, pca_amino)

    # Process target amino_kmers
    target_amino_kmers = [col for col in features if is_family_amino_kmer(col, isTarget=True)]
    with open(path.join(pca_path, f'target_amino_kmers_pca_{mode}.pkl'), 'rb') as handle:
        pca_target = pickle.load(handle)
    df_data = family_sparser(df_data, 'target_amino_kmers', target_amino_kmers, mode, pca_target)

    # Process additional feature families by prefix
    family_prefixes = [
        'local_mfe_1', 'local_dmfe_1', 'local_mfe_2', 'local_dmfe_2',
        'CAI_all_win', 'CAI_he_win', 'RCA_all_win', 'RCA_he_win',
        'tAI_win', 'RCBS_win', 'target_energy_1', 'target_energy_2',
        'target_CAI_all_win', 'target_CAI_he_win', 'target_RCA_all_win',
        'target_RCA_he_win', 'target_tAI_win', 'target_RCBS_win'
    ]
    for prefix in family_prefixes:
        fam_cols = [col for col in features if is_family_by_start(col, prefix)]
        with open(path.join(pca_path, f'{prefix}_{mode}.pkl'), 'rb') as handle:
            pca_family = pickle.load(handle)
        df_data = family_sparser(df_data, prefix, fam_cols, mode, pca_family)
    return df_data


def main(path_data, path_models, output_path, model='XGB', model_train_data='all', knn_model_split=0):
    """Load feature data, apply scaling and PCA, predict using pre-trained models, and save results."""
    df_data = pd.read_csv(path_data, index_col=0, delimiter='\t')
    df_data = df_data.select_dtypes(include='number').fillna(0)
    df_predictions = pd.DataFrame(index=df_data.index)
    os.makedirs(output_path, exist_ok=True)
    start_time = datetime.now()
    start_range, end_range = 10, 210

    for seed in range(start_range, end_range, 10):
        print(f"starting {seed}")
        seed_dir = path.join(path_models, f"train_on_{model_train_data}_curr_seed_{seed}")
        sample_path = path.join(seed_dir, 'scaler', 'scaler_sample_data.csv')
        df_sample = pd.read_csv(sample_path, index_col=0)
        sample_cols = [col for col in df_sample.columns if col not in ['CV_group', 'label']]
        df_seed = df_data[sample_cols]
        scaler_path = path.join(seed_dir, 'scaler', f"scaler_{seed}.pkl")
        with open(scaler_path, "rb") as handle:
            scaler = pickle.load(handle)
        df_scaled = pd.DataFrame(scaler.transform(df_seed), columns=df_seed.columns, index=df_seed.index)
        df_scaled.to_csv(path.join(output_path, "data_scaled.csv"))
        
        if model == "XGB":
            df_pca = generate_PCA_data(df_scaled, seed_dir, model)
            df_pca.to_csv(path.join(output_path, "data_pca.csv"))
        else:
            df_pca = df_scaled.copy()

        if model == "XGB":
            predictor = xgb.Booster()
            model_path = path.join(seed_dir, "model.json")
            predictor.load_model(model_path)
            y_pred = predictor.predict(xgb.DMatrix(df_pca.values))
        else:
            model_path = path.join(seed_dir, f"L2_{model}_model.pkl")
            with open(model_path, "rb") as handle:
                predictor = pickle.load(handle)
            y_pred = predictor.predict(df_pca.values)
        df_predictions[f"prediction_{seed}"] = y_pred
        elapsed = int((datetime.now() - start_time).total_seconds())
        print(f"Seconds since start: {elapsed}")

    df_predictions["prediction_mean"] = df_predictions.mean(axis=1)
    df_predictions.to_csv(path.join(output_path, f"predictions_{model}.csv"))
    return


if __name__ == "__main__":
    curr_dataset = "yeast_insulin"
    output_path = path.join("data", "processed_data")
    path_data = path.join("A_feature_generation", "Output", f"{curr_dataset}.csv")
    for model in ["XGB", "KNN"]:
        path_models = f"lean_{model}"
        main(path_data, path_models, output_path, model=model)
