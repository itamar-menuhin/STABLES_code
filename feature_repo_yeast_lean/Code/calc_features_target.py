#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_target.py - Target-specific feature calculation module

This module calculates features related to target gene sequences for gene expression
optimization and comparison. It compares a target gene sequence with endogenous sequences
using multiple distance metrics and integrates additional features from other modules
(CUB, sORF, sequence-based, disorder, chemical).

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

# Import local modules (reused functions)
from .utils import calc_ATG_PSSM
from .calc_features_CUB import calc_CUB
from .calc_features_sORF import calc_sORF
from .calc_features_seq import calc_nuc_fraction, calc_AA_kmers
from .calc_features_disorder import calc_disorder
from .calc_features_chemical import calc_chemical

# Constants
CODON_TABLE_PATH = '../Data/codon_tables.pkl'
DISTANCE_TYPES = ["L2", "L1", "spearman", "pearson", "KS"]

with open(CODON_TABLE_PATH, 'rb') as handle:
    CODON_TABLE = pickle.load(handle)[0]


def calculate_target_features(features, target_sequence):
    """
    Calculate target-specific features for a given target gene.
    
    The function computes codon and amino acid frequency vectors for the target,
    compares them with each endogenous gene using various distance metrics,
    and aggregates additional features from other modules.
    
    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing endogenous gene sequences (must include an 'ORF' column).
    target_sequence : str
        The target gene nucleotide sequence.
        
    Returns
    -------
    pd.DataFrame
        Updated features DataFrame with target-specific features added.
    """
    print("Calculating target-specific features...")
    # Calculate target frequencies using a helper defined below.
    target_codon_freq, target_aa_freq = calculate_frequencies(target_sequence)
    
    # Compute frequencies for endogenous sequences using the same function.
    features["codon_freq"], features["aa_freq"] = zip(*features["ORF"].apply(calculate_frequencies))
    
    # Calculate distances between target and endogenous properties.
    for property_name, target_prop in zip(["codon_freq", "aa_freq"],
                                          [target_codon_freq, target_aa_freq]):
        for distance_type in DISTANCE_TYPES:
            features[f"{property_name}_{distance_type}"] = features[property_name].apply(
                lambda prop: calculate_distance(target_prop, prop, distance_type)
            )
    
    # Integrate additional features (order preserved exactly as in the original pipeline)
    for func in [calc_CUB, calc_sORF, calc_nuc_fraction, calc_AA_kmers, calc_disorder, calc_chemical]:
        features = func(features)
    
    return features


def calculate_frequencies(sequence):
    """
    Calculate codon and amino acid frequencies for a nucleotide sequence.
    
    If the sequence ends with a stop codon ("TAG", "TGA", or "TAA"), it is removed.
    The nucleotide sequence is then translated, and frequencies are computed using CODON_TABLE.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence to analyze.
        
    Returns
    -------
    tuple
        (codon_frequencies, amino_acid_frequencies)
    """
    # Remove stop codon if present
    if sequence[-3:] in ["TAG", "TGA", "TAA"]:
        sequence = sequence[:-3]
    
    seq_obj = Seq(sequence)
    amino_acid_seq = str(seq_obj.translate(to_stop=True))
    
    # Calculate codon frequencies using the predefined codon table.
    codon_frequencies = []
    for codon_list in CODON_TABLE.values():
        for codon in codon_list:
            count = sequence.count(codon)
            codon_frequencies.append(count / (len(sequence) // 3))
    
    # Calculate amino acid frequencies.
    amino_acid_frequencies = [amino_acid_seq.count(aa) / len(amino_acid_seq) for aa in CODON_TABLE.keys()]
    
    return codon_frequencies, amino_acid_frequencies


def calculate_distance(vector1, vector2, distance_type):
    """
    Calculate the distance between two vectors using various metrics.
    
    Supported metrics:
      - "L2": Euclidean norm
      - "L1": Manhattan norm
      - "spearman": Spearman correlation
      - "pearson": Pearson correlation (using statistic attribute)
      - "KS": Kolmogorov-Smirnov statistic
    
    Parameters
    ----------
    vector1, vector2 : list or array-like
        The frequency vectors to compare.
    distance_type : str
        One of "L2", "L1", "spearman", "pearson", or "KS".
        
    Returns
    -------
    float
        The computed distance.
        
    Raises
    ------
    ValueError
        If an invalid distance type is provided.
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    if distance_type == "L2":
        return np.linalg.norm(v1 - v2, ord=2)
    elif distance_type == "L1":
        return np.linalg.norm(v1 - v2, ord=1)
    elif distance_type == "spearman":
        return spearmanr(v1, v2).correlation
    elif distance_type == "pearson":
        return pearsonr(v1, v2).statistic
    elif distance_type == "KS":
        return ks_2samp(v1, v2).statistic
    else:
        raise ValueError(f"Invalid distance type: {distance_type}")


def calculate_initiation_features(sequence):
    """
    Calculate initiation-related features for a sequence.
    
    This function analyzes the start codon context of the provided sequence. The sequence must start with ATG.
    It computes a PSSM score for the initiation region using the calc_ATG_PSSM function.
    
    Parameters
    ----------
    sequence : str
        Nucleotide sequence (must start with ATG).
        
    Returns
    -------
    dict
        A dictionary with initiation-related features.
        
    Raises
    ------
    ValueError:
        If the sequence does not start with ATG.
    """
    sequence = sequence.upper()
    if not sequence.startswith("ATG"):
        raise ValueError("Sequence must start with ATG")
    
    window_size = 30  # In codons
    atg_positions = [m.start() for m in re.finditer("ATG", sequence) if m.start() % 3 == 0]
    filtered_positions = [
        pos for pos in atg_positions
        if (pos // 3) < window_size and (pos + 5) < len(sequence)
    ]
    
    pssm_matrix = calc_ATG_PSSM()
    pssm_scores = [
        np.prod([pssm_matrix[i][sequence[pos + i]] for i in range(3)])
        for pos in filtered_positions
    ]
    
    return {
        "ATG_ORF": max(len(atg_positions) - 1, 0),
        f"ATG_ORF_window{window_size}": max(len(filtered_positions) - 1, 0),
        f"ATG_ORF_window{window_size}_mean": np.mean(pssm_scores) if pssm_scores else 0,
    }
