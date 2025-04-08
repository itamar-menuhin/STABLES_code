#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_CUB.py - Codon Usage Bias feature calculation module

This module calculates various codon usage bias metrics that characterize
translation efficiency and gene expression levels. It implements multiple
established metrics including CAI (Codon Adaptation Index), tAI (tRNA Adaptation
Index), ENC (Effective Number of Codons), and RCBS (Relative Codon Bias Strength).

The key features calculated include:
- CAI: Codon Adaptation Index for both all genes and highly expressed genes
- tAI: tRNA Adaptation Index
- ENC: Effective Number of Codons
- RCBS: Relative Codon Bias Strength
- Window-based analysis of codon bias across gene regions

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

# Constants - DO NOT CHANGE - preserves output alignment
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
        A tuple containing:
        - amino_acid_to_codon (dict): Mapping from amino acids to their codons
        - cub_weights (dict): Dictionary of different codon bias weights schemes
    """
    with open(CODON_TABLES_PATH, "rb") as handle:
        amino_acid_to_codon, _ = pickle.load(handle)

    with open(CUB_WEIGHTS_PATH, "rb") as handle:
        cub_weights = pickle.load(handle)

    return amino_acid_to_codon, cub_weights


def calculate_effective_number_of_codons(codons, amino_acid_to_codon):
    """
    Calculate the Effective Number of Codons (ENC) for a sequence.
    
    ENC is a metric that quantifies how biased a gene is in its codon
    usage, ranging from 20 (maximum bias) to 61 (no bias).
    
    Parameters
    ----------
    codons : list
        List of codons from the sequence
    amino_acid_to_codon : dict
        Mapping from amino acids to their codons
        
    Returns
    -------
    float
        The ENC value (between 20 and 61)
    """
    # Calculate homozygosity for each amino acid
    f_values = {}
    for aa, synonymous_codons in amino_acid_to_codon.items():
        if len(synonymous_codons) <= 1:
            continue  # Skip amino acids with only one codon
        
        # Count occurrences of each codon for this amino acid
        codon_counts = {codon: codons.count(codon) for codon in synonymous_codons}
        total_count = sum(codon_counts.values())
        
        if total_count == 0:
            continue
            
        # Calculate homozygosity - CRITICAL: Use same formula as original
        f_values[aa] = sum((count/total_count)**2 for count in codon_counts.values())
    
    # Group amino acids by degeneracy class (how many synonymous codons)
    degeneracy_classes = {2: [], 3: [], 4: [], 6: []}
    for aa, synonymous_codons in amino_acid_to_codon.items():
        n_codons = len(synonymous_codons)
        if n_codons in degeneracy_classes and aa in f_values:
            degeneracy_classes[n_codons].append(f_values[aa])
    
    # Calculate average homozygosity for each degeneracy class
    averages = {n: np.mean(vals) if vals else 1.0 for n, vals in degeneracy_classes.items()}
    
    # Calculate ENC - CRITICAL: Use same formula and clamping as original
    enc = 2 + 9/averages[2] + 1/averages[3] + 5/averages[4] + 3/averages[6]
    return min(61, max(20, enc))  # Clamp between 20 and 61


def calculate_RCBS(sequence):
    """
    Calculate the Relative Codon Bias Strength (RCBS) for a sequence.
    
    RCBS measures how much the codon usage deviates from what would be expected
    based on nucleotide composition alone.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence
        
    Returns
    -------
    float
        The RCBS value
    """
    # Extract codons from sequence
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    
    # Calculate nucleotide distribution at each codon position
    nt_dist = {}
    for n in range(3):
        curr_dist = {}
        for codon in codons:
            if len(codon) > n:  # Make sure codon is long enough
                nt = codon[n]
                curr_dist[nt] = curr_dist.get(nt, 0) + 1
        
        # Normalize to get probabilities
        total = sum(curr_dist.values())
        for nt in curr_dist:
            curr_dist[nt] /= total
            
        nt_dist[n] = curr_dist
    
    # Calculate RCBS for each codon - CRITICAL: Use same formula as original
    rcbs_values = []
    for codon in codons:
        if len(codon) == 3:  # Ensure we have complete codons
            # Expected frequency based on nucleotide composition
            expected_freq = np.prod([nt_dist[n].get(codon[n], 0) for n in range(3)])
            if expected_freq == 0:
                continue
                
            # Observed frequency
            observed_freq = codons.count(codon) / len(codons)
            
            # RCBS formula
            rcbs_values.append(1 + (observed_freq - expected_freq) / expected_freq)
    
    # Calculate geometric mean and adjust - CRITICAL: Use same adjustment as original
    return geometric_mean(rcbs_values) - 1 if rcbs_values else 0


def calculate_codon_usage_bias(sequence, amino_acid_to_codon, cub_weights):
    """
    Calculate comprehensive codon usage bias features for a sequence.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence
    amino_acid_to_codon : dict
        Mapping from amino acids to their codons
    cub_weights : dict
        Dictionary of different codon bias weights schemes
        
    Returns
    -------
    dict
        Dictionary of codon usage bias features
    """
    # Extract codons from sequence
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    
    # Initialize features dictionary
    cub_features = {"ENC": calculate_effective_number_of_codons(codons, amino_acid_to_codon)}
    
    # Calculate various codon usage bias metrics
    # CRITICAL: iterate through weights in same order as original
    for score_name, weights in cub_weights.items():
        # Calculate scores for all codons
        scores = [weights.get(codon, 0) for codon in codons if codon in weights and weights[codon] > 0]
        
        if not scores:
            cub_features[score_name] = 0
            cub_features[f"{score_name}_std"] = 0
            cub_features[f"{score_name}_f{FIRST_LAST}"] = 0
            cub_features[f"{score_name}_l{FIRST_LAST}"] = 0
            continue
            
        # Overall score (geometric mean) - CRITICAL: Use geometric_mean exactly as original
        cub_features[score_name] = geometric_mean(scores)
        
        # Standard deviation - CRITICAL: Use np.std exactly as original
        cub_features[f"{score_name}_std"] = np.std(scores)
        
        # First N codons score
        first_scores = scores[:min(FIRST_LAST, len(scores))]
        cub_features[f"{score_name}_f{FIRST_LAST}"] = geometric_mean(first_scores) if first_scores else 0
        
        # Last N codons score
        last_scores = scores[-min(FIRST_LAST, len(scores)):]
        cub_features[f"{score_name}_l{FIRST_LAST}"] = geometric_mean(last_scores) if last_scores else 0
    
    # Calculate RCBS metrics - CRITICAL: Use same window sizes as original
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
        The gene data with ORF sequences
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with codon usage bias features
    """
    print("Calculating codon usage bias features...")
    
    # Load required data
    amino_acid_to_codon, cub_weights = load_codon_tables_and_weights()
    
    # Apply calculation to each sequence
    for gene in features.index:
        seq = features.loc[gene, "ORF"]
        cub_dict = calculate_codon_usage_bias(seq, amino_acid_to_codon, cub_weights)
        
        # Add calculated features to DataFrame
        for key, value in cub_dict.items():
            features.loc[gene, key] = value
    
    return features
