#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_chimera.py - Chimera algorithm feature calculation module

This module implements the Chimera algorithm to calculate sequence similarity features
between genes. This algorithm measures how segments of a gene sequence match other 
sequences in the genome, which can impact synthetic gene expression levels.

The key feature calculated is:
- chimeraARS: A score reflecting the average length of shared subsequences
  with other genes in the genome

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Yofe I, et al. Accurate, model-based tuning of synthetic gene expression 
        using introns in S. cerevisiae. Bioinformatics. 2015;31(8):1161-1168.
        doi:10.1093/bioinformatics/btu775
"""

import pandas as pd
import numpy as np
import os
from bisect import bisect_left

# Define constants - preserving exact paths for output consistency
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
SUFFIX_ARRAY_PATH = os.path.join(DATA_DIR, "suffix_array.csv")


def score_current_sequence(sequence, suffix_list):
    """
    Calculate the length of the longest common prefix between a sequence
    and any sequence in the suffix list.
    
    This function uses binary search to efficiently find matches in the suffix array.
    
    Parameters
    ----------
    sequence : str
        The sequence to score
    suffix_list : list
        List of suffixes from the suffix array
        
    Returns
    -------
    int
        The score for the sequence (length of longest common prefix)
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
    Calculate the chimera score for a gene.
    
    This function computes sequence similarity between a gene and other 
    genes in the genome by measuring the average length of shared subsequences.
    
    Parameters
    ----------
    sequence : str
        The full sequence of the gene
    current_orf : str
        The ORF (Open Reading Frame) identifier of the gene
    suffix_array : pandas.DataFrame
        The suffix array DataFrame containing all gene suffixes
        
    Returns
    -------
    float
        The chimera score for the gene (average length of shared subsequences)
    """
    # Filter out suffixes from the current ORF to avoid self-matching
    suffix_list = suffix_array.loc[suffix_array["ORF"] != current_orf, "seq"].tolist()
    
    # Calculate average score across all possible starting positions
    return np.mean([score_current_sequence(sequence[i:], suffix_list) for i in range(len(sequence))])


def calculate_chimera_features(features):
    """
    Add chimera algorithm features to the features DataFrame.
    
    This function applies the Chimera algorithm to calculate sequence similarity 
    scores for each gene in the dataset. This feature helps predict synthetic gene 
    expression by measuring how segments of the sequence match other genes in the 
    genome, as demonstrated in Yofe et al. (2015).
    
    Parameters
    ----------
    features : pandas.DataFrame
        The input features DataFrame containing gene sequences
        
    Returns
    -------
    pandas.DataFrame
        The updated features DataFrame with chimera features
        
    Raises
    ------
    FileNotFoundError
        If the suffix array file is not found
    """
    if not os.path.isfile(SUFFIX_ARRAY_PATH):
        raise FileNotFoundError(f"Suffix array file not found: {SUFFIX_ARRAY_PATH}")

    suffix_array = pd.read_csv(SUFFIX_ARRAY_PATH)
    print("Calculating chimera features...")

    # CRITICAL: Preserving exact calculation method to ensure identical output
    features["chimeraARS"] = features.apply(
        lambda row: score_gene(row["ORF"], row.name, suffix_array), axis=1
    )

    print("Chimera features calculation complete.")
    return features