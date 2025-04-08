#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_lfe.py - Local Folding Energy feature calculation module

This module calculates features related to the local folding energies of
RNA structures that form in coding sequences. These structures can influence
translation efficiency, RNA stability, and protein expression levels.

The key features calculated include:
- Local Minimum Free Energy (MFE) across sequence windows
- Delta MFE compared to randomized sequences (Z-score)
- Average local folding energy metrics

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Tuller T, Waldman YY, Kupiec M, Ruppin E. Translation efficiency is
        determined by both codon bias and folding energy. Proc Natl Acad Sci USA. 2010.
    [2] Kudla G, Murray AW, Tollervey D, Plotkin JB. Coding-sequence
        determinants of gene expression in Escherichia coli. Science. 2009.
"""

from Bio.Seq import Seq
import ViennaRNA as VRNA
import random
from statistics import fmean
import pandas as pd

# Constants - DO NOT CHANGE - preserves output alignment
LOCAL_WINDOW_SIZE = 40  # Size of window for local folding (in nucleotides)
STEP_SIZE = 10          # Step size for sliding windows (in nucleotides)


def calculate_mfe(sequence):
    """
    Calculate the Minimum Free Energy (MFE) of an RNA sequence using ViennaRNA.
    
    The MFE value indicates the thermodynamic stability of RNA structures, with
    more negative values indicating stronger folding propensity.

    Parameters
    ----------
    sequence : str
        The RNA sequence
        
    Returns
    -------
    float
        The MFE value in kcal/mol
    """
    rna_sequence = str(Seq(sequence).transcribe())  # Transcribe DNA to RNA
    return VRNA.fold(rna_sequence)[1]


def generate_random_permuted_sequence(dna_sequence):
    """
    Generate a random permuted DNA sequence while preserving amino acid translation.
    
    This function preserves codon composition but disrupts the original sequence order,
    allowing for comparison between the original sequence and a sequence with identical
    codon usage but different ordering.

    Parameters
    ----------
    dna_sequence : str
        The DNA sequence
        
    Returns
    -------
    str
        The permuted DNA sequence with identical codon usage
    """
    # CRITICAL: Preserve exact implementation to maintain output consistency
    codons = [dna_sequence[i:i + 3] for i in range(0, len(dna_sequence), 3)]
    random.shuffle(codons)
    return ''.join(codons)


def calculate_local_mfe_features(dna_sequence):
    """
    Calculate local MFE and delta MFE features for a DNA sequence.
    
    This function calculates MFE in sliding windows and compares to
    a randomly permuted sequence to calculate delta MFE values. These
    delta values help identify regions where the specific sequence order
    (rather than just composition) contributes to RNA structure.

    Parameters
    ----------
    dna_sequence : str
        The DNA sequence
        
    Returns
    -------
    dict
        Dictionary containing local folding energy features
    """
    # Generate a randomly permuted sequence with the same codon composition
    # CRITICAL: Use the same random sequence generation method as original
    random_sequence = generate_random_permuted_sequence(dna_sequence)
    mfe_features = {}

    # Calculate local MFE and delta MFE in sliding windows
    # CRITICAL: Use the exact same window parameters and step sizes as original
    for start_index in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE):
        window = dna_sequence[start_index:start_index + LOCAL_WINDOW_SIZE]
        random_window = random_sequence[start_index:start_index + LOCAL_WINDOW_SIZE]

        true_mfe = calculate_mfe(window)
        random_mfe = calculate_mfe(random_window)

        # CRITICAL: Use the same feature names as original
        mfe_features[f'local_mfe_{start_index}'] = true_mfe
        mfe_features[f'local_delta_mfe_{start_index}'] = true_mfe - random_mfe

    # Calculate averages across all windows
    # CRITICAL: Use the exact same calculation method for consistency
    local_mfe_values = [
        mfe_features[f'local_mfe_{start_index}'] 
        for start_index in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE)
    ]
    
    local_delta_mfe_values = [
        mfe_features[f'local_delta_mfe_{start_index}'] 
        for start_index in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE)
    ]

    # CRITICAL: Use the same feature names and calculation method
    mfe_features['average_local_mfe'] = fmean(local_mfe_values) if local_mfe_values else 0
    mfe_features['average_local_delta_mfe'] = fmean(local_delta_mfe_values) if local_delta_mfe_values else 0

    return mfe_features


def calculate_LFE_features(features):
    """
    Calculate local folding energy (LFE) features for each gene.
    
    LFE features are important predictors of translation efficiency, as
    stable RNA structures can impede ribosome progression. This function
    computes various LFE metrics for multiple windows along each gene.

    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with 'ORF' column containing nucleotide sequences
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with LFE features
    """
    print("Calculating local folding energy features...")
    
    # Apply LFE calculation to each ORF
    # CRITICAL: Use the exact same calculation method as original
    lfe_data = features['ORF'].apply(calculate_local_mfe_features)
    
    # Convert the resulting dictionaries to a DataFrame and handle NaN values
    # CRITICAL: Use the same fillna value (0) as original
    lfe_data_normalized = pd.DataFrame(lfe_data.tolist()).fillna(0)
    
    # Join with original features
    # CRITICAL: Use the same method for joining DataFrames
    return pd.concat([features, lfe_data_normalized.set_index(features.index)], axis=1)



