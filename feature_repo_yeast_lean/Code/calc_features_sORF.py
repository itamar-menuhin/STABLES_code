#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_sORF.py - Shifted Open Reading Frame (sORF) feature calculation module

This module calculates features related to shifted Open Reading Frames (sORFs) within gene
sequences. Shifted ORFs represent alternative reading frames that differ from the primary 
reading frame and can impact translation efficiency through mechanisms like ribosome 
frameshifting.

The key features calculated include:
- Number of shifted ORFs in the sequence
- Maximum and mean shifted ORF lengths
- Window-based analysis of shifted ORF presence and characteristics

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Ketteler R. On programmed ribosomal frameshifting: the alternative proteomes.
        Front Genet. 2012.
    [2] Dinman JD. Mechanisms and implications of programmed translational frameshifting.
        Wiley Interdiscip Rev RNA. 2012.
"""

import pandas as pd
from Bio.Seq import Seq

# Define sORF window sizes in codons - DO NOT CHANGE - preserves output alignment
SORF_WINDOWS = [30, 200]


def calculate_sORF(sequence):
    """
    Calculate shifted Open Reading Frame (sORF) features for a sequence.
    
    This function identifies potential alternative reading frames that are
    shifted from the primary reading frame of the gene and calculates
    various metrics related to these shifted ORFs.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence to analyze
        
    Returns
    -------
    dict
        Dictionary containing shifted ORF-related features
    """
    seq_obj = Seq(sequence)
    
    # Find all potential start codons (ATG)
    start_positions = [i for i in range(len(seq_obj)) if seq_obj[i:i + 3] == "ATG"]
    
    # Find all stop codons (TAG, TAA, TGA)
    stop_positions = [
        i for i in range(len(seq_obj))
        if seq_obj[i:i + 3] in {"TAG", "TAA", "TGA"}
    ]
    
    # Calculate valid ORF lengths (must be multiple of 3 from start to stop)
    orf_lengths = [
        stop - start + 3
        for start in start_positions
        for stop in stop_positions
        if stop > start and (stop - start) % 3 == 0
    ]
    
    # Calculate global shifted ORF features
    sORF_features = {
        "num_sORF": len(orf_lengths),
        "max_sORF_len": max(orf_lengths, default=0),
        "mean_sORF_len": sum(orf_lengths) / len(orf_lengths) if orf_lengths else 0,
    }
    
    # Calculate shifted ORF features within defined windows
    # CRITICAL: The window_orfs creation logic must be preserved exactly as in original code
    for window in SORF_WINDOWS:
        window_orfs = [
            length for start, length in zip(start_positions, orf_lengths) 
            if start <= window * 3
        ]
        
        sORF_features[f"num_sORF_win{window}"] = len(window_orfs)
        sORF_features[f"max_sORF_win{window}"] = max(window_orfs, default=0)
        sORF_features[f"mean_sORF_win{window}"] = sum(window_orfs) / len(window_orfs) if window_orfs else 0
    
    return sORF_features


def calculate_sORF_features(features):
    """
    Calculate shifted ORF (sORF) features for each gene in the dataset.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with an 'ORF' column containing nucleotide sequences
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with shifted ORF features
    """
    print("Calculating shifted ORF features...")
    
    # Apply calculation to each sequence - CRITICAL: maintain exact column creation order
    sORF_features_df = features["ORF"].apply(calculate_sORF).apply(pd.Series)
    
    # Combine with original features
    return pd.concat([features, sORF_features_df], axis=1)
