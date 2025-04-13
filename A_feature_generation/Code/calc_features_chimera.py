#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_chimera.py - Chimera algorithm feature calculation module

This module implements the Chimera algorithm to compute sequence similarity features 
for genes. The key feature is chimeraARS, which reflects the average length of shared 
subsequences (excluding self-matches) across the genome.

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]

References:
    [1] Yofe I, et al. Accurate, model-based tuning of synthetic gene expression 
        using introns in S. cerevisiae. Bioinformatics. 2015;31(8):1161-1168.
        doi:10.1093/bioinformatics/btu775
"""

import os
import pandas as pd
import numpy as np
from bisect import bisect_left

# Preserve exact paths for consistency
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
SUFFIX_ARRAY_PATH = os.path.join(DATA_DIR, "suffix_array.csv")


def score_current_sequence(sequence, suffix_list):
    """
    Calculate the longest common prefix length between the sequence and any suffix in suffix_list.
    
    Parameters
    ----------
    sequence : str
        Sequence to score.
    suffix_list : list of str
        List of suffixes.
        
    Returns
    -------
    int
        Longest common prefix length.
    """
    score = 0
    for i in range(len(sequence)):
        prefix = sequence[:i + 1]
        pos = bisect_left(suffix_list, prefix)
        if pos < len(suffix_list) and suffix_list[pos].startswith(prefix):
            score += 1
        else:
            break
    return score


def score_gene(sequence, current_orf, suffix_array):
    """
    Calculate the chimera score for a gene by averaging the longest common prefix 
    across all starting positions in the gene, excluding self-matching suffixes.
    
    Parameters
    ----------
    sequence : str
        Full gene sequence.
    current_orf : str
        Identifier for the gene's ORF (used to exclude self-matches).
    suffix_array : pd.DataFrame
        DataFrame with a "seq" column containing gene suffixes and "ORF" column.
        
    Returns
    -------
    float
        The average longest common prefix score.
    """
    # Exclude suffixes from the current ORF to avoid self-matches
    suffix_list = suffix_array.loc[suffix_array["ORF"] != current_orf, "seq"].tolist()
    # Calculate score for each starting position and average
    scores = [score_current_sequence(sequence[i:], suffix_list) for i in range(len(sequence))]
    return np.mean(scores)


def calculate_chimera_features(features):
    """
    Compute the chimeraARS feature for each gene and add it to the features DataFrame.
    
    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing gene sequences (with column "ORF").
        
    Returns
    -------
    pd.DataFrame
        Updated DataFrame with a new "chimeraARS" column.
        
    Raises
    ------
    FileNotFoundError
        If the suffix array file is not found.
    """
    if not os.path.isfile(SUFFIX_ARRAY_PATH):
        raise FileNotFoundError(f"Suffix array file not found: {SUFFIX_ARRAY_PATH}")
    
    suffix_array = pd.read_csv(SUFFIX_ARRAY_PATH)
    print("Calculating chimera features...")
    
    # CRITICAL: Preserve the exact calculation method to ensure identical output
    features["chimeraARS"] = features.apply(
        lambda row: score_gene(row["ORF"], row.name, suffix_array), axis=1
    )
    
    print("Chimera features calculation complete.")
    return features