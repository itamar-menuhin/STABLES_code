#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_target.py - Target-specific feature calculation module

This module calculates features related to target gene sequences for gene expression
optimization and comparison. It implements functions to compare a target gene sequence
with endogenous sequences using multiple distance metrics, and aggregates additional
features from other modules.

The key features calculated include:
- Codon frequency distances (L1, L2, Spearman, Pearson, KS)
- Amino acid frequency distances
- Integration with other feature calculation modules for comprehensive analysis

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
"""

import pickle
import re
from Bio.Seq import Seq
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp
import os

# Import local modules - Use the same import pattern as original code
from .utils import calc_ATG_PSSM
from .calc_features_CUB import calc_CUB
from .calc_features_sORF import calc_sORF
from .calc_features_seq import calc_nuc_fraction, calc_AA_kmers
from .calc_features_disorder import calc_disorder
from .calc_features_chemical import calc_chemical

# Constants - DO NOT CHANGE - preserves output alignment
CODON_TABLE_PATH = '../Data/codon_tables.pkl'
DISTANCE_TYPES = ["L2", "L1", "spearman", "pearson", "KS"]

# Load codon table
with open(CODON_TABLE_PATH, 'rb') as handle:
    CODON_TABLE = pickle.load(handle)[0]


def calculate_target_features(features, target_sequence):
    """
    Calculate target-specific features for a given target gene.
    
    This function compares the target gene sequence properties with endogenous 
    gene sequences and calculates various distance metrics between them,
    which can be used for gene expression optimization.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The input features DataFrame containing endogenous gene sequences.
        Must contain an 'ORF' column.
    target_sequence : str
        The target gene nucleotide sequence to compare against.
        
    Returns
    -------
    pandas.DataFrame
        The updated features DataFrame with target-specific features added.
    """
    print("Calculating target-specific features...")

    # Calculate target properties
    target_codon_freq, target_aa_freq = calculate_frequencies(target_sequence)
    
    # CRITICAL: This creates two temporary columns that must be computed exactly as original
    features["codon_freq"], features["aa_freq"] = zip(*features["ORF"].apply(calculate_frequencies))

    # Calculate distances between target and endogenous properties
    # CRITICAL: The property_name and distance_name order must be preserved
    for property_name, target_property in zip(["codon_freq", "aa_freq"], [target_codon_freq, target_aa_freq]):
        for distance_name in DISTANCE_TYPES:
            features[f"{property_name}_{distance_name}"] = features[property_name].apply(
                lambda prop: calculate_distance(target_property, prop, distance_name)
            )

    # Add features from other modules
    # CRITICAL: The order of function application must be preserved exactly
    for func in [calc_CUB, calc_sORF, calc_nuc_fraction, calc_AA_kmers, calc_disorder, calc_chemical]:
        features = func(features)

    return features


def calculate_frequencies(sequence):
    """
    Calculate codon and amino acid frequencies for a sequence.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence to analyze.
        
    Returns
    -------
    tuple
        (codon_frequencies, amino_acid_frequencies) as lists of frequency values.
    """
    # CRITICAL: Preserve original stop codon handling
    sequence = sequence[:-3] if sequence[-3:] in ["TAG", "TGA", "TAA"] else sequence
    
    # Translate to amino acid sequence
    seq_obj = Seq(sequence)
    amino_acid_seq = str(seq_obj.translate(to_stop=True))

    # Calculate codon frequencies - CRITICAL: Use exact same calculation as original
    codon_frequencies = []
    for codon_list in CODON_TABLE.values():
        for codon in codon_list:
            count = sequence.count(codon)
            codon_frequencies.append(count / (len(sequence) // 3))

    # Calculate amino acid frequencies - CRITICAL: Use exact same calculation as original
    amino_acid_frequencies = [amino_acid_seq.count(aa) / len(amino_acid_seq) 
                             for aa in CODON_TABLE.keys()]

    return codon_frequencies, amino_acid_frequencies


def calculate_distance(vector1, vector2, distance_type):
    """
    Calculate the distance between two vectors using various metrics.
    
    Parameters
    ----------
    vector1 : list or array-like
        First vector to compare.
    vector2 : list or array-like
        Second vector to compare.
    distance_type : str
        Type of distance metric to use. Must be one of:
        "L2" (Euclidean), "L1" (Manhattan), "spearman", "pearson", or "KS".
        
    Returns
    -------
    float
        The calculated distance value.
        
    Raises
    ------
    ValueError
        If an invalid distance type is specified.
    """
    if distance_type == "L2":
        return np.linalg.norm(np.array(vector1) - np.array(vector2), ord=2)
    elif distance_type == "L1":
        return np.linalg.norm(np.array(vector1) - np.array(vector2), ord=1)
    elif distance_type == "spearman":
        return spearmanr(vector1, vector2).correlation
    elif distance_type == "pearson":
        return pearsonr(vector1, vector2).statistic
    elif distance_type == "KS":
        return ks_2samp(vector1, vector2).statistic
    raise ValueError(f"Invalid distance type: {distance_type}")


def calculate_initiation_features(sequence):
    """
    Calculate initiation-related features for a sequence.
    
    These features characterize the translation initiation context
    by analyzing ATG codons and their surrounding sequences.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence, must start with ATG.
        
    Returns
    -------
    dict
        A dictionary of initiation-related features.
        
    Raises
    ------
    ValueError
        If the sequence does not start with ATG.
    """
    sequence = sequence.upper()
    if not sequence.startswith("ATG"):
        raise ValueError("Sequence must start with ATG")

    # CRITICAL: Use the same window size as original code
    window_size = 30  # In codons
    
    atg_positions = [match.start() for match in re.finditer("ATG", sequence) if match.start() % 3 == 0]
    filtered_positions = [pos for pos in atg_positions if (pos // 3) < window_size and (pos + 5) < len(sequence)]

    pssm_matrix = calc_ATG_PSSM()
    
    # CRITICAL: Calculate PSSM scores exactly as in original code
    pssm_scores = [
        np.prod([pssm_matrix[i][sequence[pos + i]] for i in range(3)]) 
        for pos in filtered_positions
    ]

    # CRITICAL: Feature names and calculation must be identical to original
    return {
        "ATG_ORF": max(len(atg_positions) - 1, 0),
        f"ATG_ORF_window{window_size}": max(len(filtered_positions) - 1, 0),
        f"ATG_ORF_window{window_size}_mean": np.mean(pssm_scores) if pssm_scores else 0,
    }
