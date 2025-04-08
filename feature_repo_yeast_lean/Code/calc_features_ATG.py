#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_ATG.py - Start codon context feature calculation module

This module calculates features related to translation initiation sites and
start codon contexts. The sequence context around the ATG start codon has
been shown to significantly influence translation efficiency across species.

The key features calculated include:
- Number of ATG codons in the UTR and ORF regions
- Position-specific scoring matrix (PSSM) scores of ATG contexts
- Analysis of ATG codons within the first 30 codons of the ORF

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Kozak M. Regulation of translation via mRNA structure in prokaryotes
        and eukaryotes. Gene. 1999;234(2):187-208.
    [2] Noderer WL, et al. Quantitative analysis of mammalian translation
        initiation sites by FACS-seq. Mol Syst Biol. 2014;10:748.
"""

import pandas as pd
import re
import os
import sys
from statistics import fmean
import pickle
import numpy as np

# Add parent directory to path for proper imports - DO NOT CHANGE - preserves import structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Code.utils import calculate_ATG_PSSM

# Constants - DO NOT CHANGE - preserves output alignment
CONTEXT_WINDOW = 150
ATG_WINDOW = 30  # Number of codons to consider for ATG-related features
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Data')


def calculate_pssm_score(sequence, pssm_matrix):
    """
    Calculate the Position-Specific Scoring Matrix (PSSM) score for a given sequence.
    
    PSSM scores quantify how well a sequence matches the consensus motif for
    translation initiation, with higher scores indicating stronger contexts.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence to score
    pssm_matrix : pandas.DataFrame
        The PSSM matrix with nucleotide weights by position
        
    Returns
    -------
    float
        The calculated PSSM score, representing the strength of the ATG context
    """
    # CRITICAL: Use exact same calculation as original
    return sum(pssm_matrix.loc[nucleotide, index] for index, nucleotide in enumerate(sequence))


def find_atg_locations(sequence, max_length=None):
    """
    Find ATG codon locations in a sequence that are in the correct reading frame.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence to analyze
    max_length : int, optional
        Maximum length to consider in the sequence
        
    Returns
    -------
    list
        List of ATG codon positions in the correct reading frame
    """
    # CRITICAL: Use exact same calculation as original
    sequence = sequence[:max_length] if max_length else sequence
    return [match.start() for match in re.finditer('ATG', sequence) if match.start() % 3 == 0]


def calculate_atg_features(orf_sequence, utr_sequence, pssm_matrix):
    """
    Calculate ATG-related features for a given ORF and UTR sequence.
    
    This function analyzes both the main ATG start codon context and alternative
    ATG codons that could affect translation efficiency through mechanisms like
    leaky scanning or reinitiation.
    
    Parameters
    ----------
    orf_sequence : str
        The Open Reading Frame (ORF) sequence
    utr_sequence : str
        The Untranslated Region (UTR) sequence
    pssm_matrix : pandas.DataFrame
        The PSSM matrix for scoring ATG contexts
        
    Returns
    -------
    dict
        A dictionary containing ATG-related features
    """
    # CRITICAL: Use exact same sequence limits as original
    orf_sequence = orf_sequence[:CONTEXT_WINDOW]
    utr_sequence = utr_sequence[-9:]

    # CRITICAL: Find ATG locations using same method as original
    utr_atg_locations = find_atg_locations(utr_sequence)
    orf_atg_locations = find_atg_locations(orf_sequence)

    # CRITICAL: Handle main ATG exactly as original
    if orf_atg_locations and orf_atg_locations[0] == 0:
        orf_atg_locations = orf_atg_locations[1:]  # Exclude the main ATG

    # CRITICAL: Calculate main ATG score using same context window
    main_atg_context = utr_sequence[-6:] + orf_sequence[:6]
    main_atg_score = calculate_pssm_score(main_atg_context, pssm_matrix)

    # CRITICAL: Calculate ATG features within the first 30 codons using same window 
    window_scores = [
        calculate_pssm_score(orf_sequence[loc - 6:loc + 6], pssm_matrix)
        for loc in orf_atg_locations if 6 <= loc <= ATG_WINDOW * 3
    ]

    # CRITICAL: Return exact same feature dictionary structure
    return {
        "ATG_UTR_count": len(utr_atg_locations),
        "ATG_ORF_count": len(orf_atg_locations),
        "ATG_main_score": main_atg_score,
        "ATG_ORF_window_count": len(window_scores),
        "ATG_ORF_window_mean_absolute": fmean(window_scores) if window_scores else 0,
        "ATG_ORF_window_mean_relative": (fmean(window_scores) - main_atg_score) if window_scores else 0,
        "ATG_ORF_window_max_absolute": max(window_scores, default=0),
        "ATG_ORF_window_max_relative": (max(window_scores, default=0) - main_atg_score) if window_scores else 0,
    }


def calculate_ATG_features(features):
    """
    Calculate ATG-related features for each gene in the dataset.
    
    This function integrates ATG context analysis for all genes, providing
    insights into potential translation efficiency based on start codon
    contexts and alternative start sites.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with sequence information
        
    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with ATG-related features
    """
    print("Calculating ATG-related features...")
    
    # CRITICAL: Get the PSSM matrix using same method as original
    pssm_matrix = calculate_ATG_PSSM()
    
    # CRITICAL: Handle missing UTR column exactly as original
    if 'UTR' not in features.columns:
        features['UTR'] = ''
    
    # CRITICAL: Apply calculation with same approach as original    
    atg_data = features.apply(
        lambda row: calculate_atg_features(row['ORF'], row.get('UTR', ''), pssm_matrix), axis=1
    )
    atg_df = pd.DataFrame(atg_data.tolist(), index=features.index)
    return pd.concat([features, atg_df], axis=1)

