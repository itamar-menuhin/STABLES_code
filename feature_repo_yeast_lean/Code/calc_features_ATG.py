#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_ATG.py - Start Codon Context Feature Calculation Module

Calculates ATG-related features including:
  - Counting ATG occurrences in UTR and ORF.
  - Scoring the main ATG context (last 6 nt of UTR + first 6 nt of ORF).
  - Scoring additional ATG sites within a fixed window.

CRITICAL: The sequence limits and window sizes (CONTEXT_WINDOW, ATG_WINDOW) are preserved.

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Kozak M. Regulation of translation via mRNA structure. Gene. 1999.
    [2] Noderer WL, et al. Mol Syst Biol. 2014.
"""

import os
import sys
import re
from statistics import fmean
import numpy as np
import pandas as pd

# Ensure proper import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Code.utils import calculate_ATG_PSSM  # Must return the pssm_matrix as in the original

# Constants (do not change)
CONTEXT_WINDOW = 150  # Use only first 150 nt of ORF
ATG_WINDOW = 30       # Only consider additional ATG sites within first 30 codons

def calculate_pssm_score(sequence, pssm_matrix):
    """
    Compute and return the PSSM score for a sequence.
    """
    # Assumes pssm_matrix is indexed by nucleotide with a column per position.
    return sum(pssm_matrix.loc[nuc, i] for i, nuc in enumerate(sequence))

def find_atg_locations(sequence, max_length=None):
    """
    Return starting positions of 'ATG' (in frame) from the sequence.
    """
    if max_length:
        sequence = sequence[:max_length]
    return [m.start() for m in re.finditer('ATG', sequence) if m.start() % 3 == 0]

def calculate_atg_features(orf_sequence, utr_sequence, pssm_matrix):
    """
    Calculate ATG-related features.
    
    The ORF is truncated to CONTEXT_WINDOW nt.
    The UTR context is taken from its last 9 nt. The main context is the last 6 nt of UTR
    concatenated with the first 6 nt of ORF.
    
    Returns a dictionary with feature values.
    """
    orf_sequence = orf_sequence[:CONTEXT_WINDOW]
    utr_sequence = utr_sequence[-9:]  # Use last 9 nt from UTR

    utr_atg = find_atg_locations(utr_sequence)
    orf_atg = find_atg_locations(orf_sequence)
    if orf_atg and orf_atg[0] == 0:
        orf_atg = orf_atg[1:]
    
    main_context = utr_sequence[-6:] + orf_sequence[:6]
    main_score = calculate_pssm_score(main_context, pssm_matrix)
    
    window_scores = [
        calculate_pssm_score(orf_sequence[loc - 6:loc + 6], pssm_matrix)
        for loc in orf_atg if 6 <= loc <= ATG_WINDOW * 3
    ]
    
    return {
        "ATG_UTR_count": len(utr_atg),
        "ATG_ORF_count": len(orf_atg),
        "ATG_main_score": main_score,
        "ATG_ORF_window_count": len(window_scores),
        "ATG_ORF_window_mean_absolute": fmean(window_scores) if window_scores else 0,
        "ATG_ORF_window_mean_relative": (fmean(window_scores) - main_score) if window_scores else 0,
        "ATG_ORF_window_max_absolute": max(window_scores, default=0),
        "ATG_ORF_window_max_relative": (max(window_scores, default=0) - main_score) if window_scores else 0,
    }

def calculate_ATG_features(features):
    """
    Apply ATG feature calculation for each gene in a DataFrame.
    
    Ensures 'UTR' column is present (defaults to empty if missing) and returns
    the DataFrame augmented with new ATG feature columns.
    """
    print("Calculating ATG-related features...")
    pssm_matrix = calculate_ATG_PSSM()
    if 'UTR' not in features.columns:
        features['UTR'] = ''
    atg_data = features.apply(
        lambda row: calculate_atg_features(row['ORF'], row.get('UTR', ''), pssm_matrix),
        axis=1
    )
    atg_df = pd.DataFrame(atg_data.tolist(), index=features.index)
    return pd.concat([features, atg_df], axis=1)

