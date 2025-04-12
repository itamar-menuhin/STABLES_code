#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_CUB.py - Codon Usage Bias feature calculation module

Calculates various codon usage bias metrics that characterize translation efficiency
and gene expression, including CAI, tAI, ENC, RCBS, and window-based analysis.

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Sharp PM, Li WH. The codon Adaptation Index--a measure of directional synonymous
        codon usage bias, and its potential applications. Nucleic Acids Res. 1987.
    [2] dos Reis M, Savva R, Wernisch L. Solving the riddle of codon usage preferences:
        a test for translational selection. Nucleic Acids Res. 2004.
    [3] Wright F. The 'effective number of codons' used in a gene. Gene. 1990.
"""

import os
import pickle
import re
from Bio.Seq import Seq
from statistics import geometric_mean
import pandas as pd
import numpy as np
from Code.utils import calculate_enc  # Reusable ENC function

# Constants - DO NOT CHANGE (preserves output alignment)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
CODON_TABLES_PATH = os.path.join(DATA_DIR, "codon_tables.pkl")
CUB_WEIGHTS_PATH = os.path.join(DATA_DIR, "CUB_weights.pkl")
FIRST_LAST = 50  # Number of first/last codons to analyze

def load_codon_tables_and_weights():
    """
    Load codon tables and CUB weights from pickle files.
    
    Returns
    -------
    tuple
        (amino_acid_to_codon, cub_weights)
    """
    with open(CODON_TABLES_PATH, "rb") as handle:
        amino_acid_to_codon, _ = pickle.load(handle)
    with open(CUB_WEIGHTS_PATH, "rb") as handle:
        cub_weights = pickle.load(handle)
    return amino_acid_to_codon, cub_weights

def calculate_RCBS(sequence):
    """
    Calculate the Relative Codon Bias Strength (RCBS) for a nucleotide sequence.
    
    Parameters
    ----------
    sequence : str
        
    Returns
    -------
    float
        The RCBS value adjusted by geometric mean.
    """
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    nt_dist = {}
    for n in range(3):
        curr_dist = {}
        for codon in codons:
            if len(codon) > n:
                nt = codon[n]
                curr_dist[nt] = curr_dist.get(nt, 0) + 1
        total = sum(curr_dist.values())
        for nt in curr_dist:
            curr_dist[nt] /= total
        nt_dist[n] = curr_dist
    
    rcbs_values = []
    for codon in codons:
        if len(codon) == 3:
            expected_freq = np.prod([nt_dist[n].get(codon[n], 0) for n in range(3)])
            if expected_freq == 0:
                continue
            observed_freq = codons.count(codon) / len(codons)
            rcbs_values.append(1 + (observed_freq - expected_freq) / expected_freq)
    return geometric_mean(rcbs_values) - 1 if rcbs_values else 0

def calculate_codon_usage_bias(sequence, amino_acid_to_codon, cub_weights):
    """
    Calculate comprehensive codon usage bias features for a nucleotide sequence.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence.
    amino_acid_to_codon : dict
        Mapping from amino acids to their codons.
    cub_weights : dict
        Dictionary of different codon bias weight schemes.
        
    Returns
    -------
    dict
        Dictionary of codon usage bias features.
    """
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    # Use calculate_enc from utils instead of a local implementation.
    cub_features = {"ENC": calculate_enc(codons, amino_acid_to_codon)}
    
    for score_name, weights in cub_weights.items():
        scores = [weights.get(codon, 0) for codon in codons if codon in weights and weights[codon] > 0]
        if not scores:
            cub_features[score_name] = 0
            cub_features[f"{score_name}_std"] = 0
            cub_features[f"{score_name}_f{FIRST_LAST}"] = 0
            cub_features[f"{score_name}_l{FIRST_LAST}"] = 0
            continue
        cub_features[score_name] = geometric_mean(scores)
        cub_features[f"{score_name}_std"] = np.std(scores)
        first_scores = scores[:min(FIRST_LAST, len(scores))]
        cub_features[f"{score_name}_f{FIRST_LAST}"] = geometric_mean(first_scores) if first_scores else 0
        last_scores = scores[-min(FIRST_LAST, len(scores)):]
        cub_features[f"{score_name}_l{FIRST_LAST}"] = geometric_mean(last_scores) if last_scores else 0
    
    cub_features["RCBS"] = calculate_RCBS(sequence)
    cub_features[f"RCBS_f{FIRST_LAST}"] = calculate_RCBS(sequence[:min(FIRST_LAST*3, len(sequence))])
    cub_features[f"RCBS_l{FIRST_LAST}"] = calculate_RCBS(sequence[-min(FIRST_LAST*3, len(sequence)):])
    
    return cub_features

def calc_CUB(features):
    """
    Calculate codon usage bias (CUB) features for each gene in the dataset.
    
    Parameters
    ----------
    features : pandas.DataFrame
        DataFrame with an "ORF" column containing gene sequences.
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with calculated CUB features.
    """
    print("Calculating codon usage bias features...")
    amino_acid_to_codon, cub_weights = load_codon_tables_and_weights()
    for gene in features.index:
        seq = features.loc[gene, "ORF"]
        cub_dict = calculate_codon_usage_bias(seq, amino_acid_to_codon, cub_weights)
        for key, value in cub_dict.items():
            features.loc[gene, key] = value
    return features
