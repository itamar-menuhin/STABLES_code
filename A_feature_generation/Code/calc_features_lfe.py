#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_lfe.py - Local Folding Energy (LFE) feature calculation module

Calculates features related to local folding energies of RNA structures in coding sequences.
Key features include:
  - Local Minimum Free Energy (MFE) in sliding windows
  - Delta MFE compared to randomized sequences (Z-score)
  - Average local MFE metrics

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Tuller T, et al. Proc Natl Acad Sci USA. 2010.
    [2] Kudla G, et al. Science. 2009.
"""

from Bio.Seq import Seq
import ViennaRNA as VRNA
import random
from statistics import fmean
import pandas as pd

# Constants - DO NOT CHANGE (preserves output alignment)
LOCAL_WINDOW_SIZE = 40  # Window size in nucleotides
STEP_SIZE = 10          # Sliding window step size

def calculate_mfe(sequence):
    """
    Calculate the Minimum Free Energy (MFE) of an RNA sequence using ViennaRNA.
    
    Parameters
    ----------
    sequence : str
        RNA sequence.
        
    Returns
    -------
    float
        MFE value in kcal/mol.
    """
    rna_seq = str(Seq(sequence).transcribe())
    return VRNA.fold(rna_seq)[1]

def generate_random_permuted_sequence(dna_sequence):
    """
    Generate a random permuted DNA sequence preserving codon composition.
    
    Parameters
    ----------
    dna_sequence : str
        DNA sequence.
        
    Returns
    -------
    str
        Permuted DNA sequence.
    """
    codons = [dna_sequence[i:i+3] for i in range(0, len(dna_sequence), 3)]
    random.shuffle(codons)
    return ''.join(codons)

def calculate_local_mfe_features(dna_sequence):
    """
    Calculate local MFE and delta MFE features using sliding windows.
    
    Parameters
    ----------
    dna_sequence : str
        DNA sequence.
        
    Returns
    -------
    dict
        Dictionary containing MFE and delta MFE for each window as well as average values.
    """
    random_seq = generate_random_permuted_sequence(dna_sequence)
    mfe_features = {}
    
    for start in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE):
        window = dna_sequence[start:start + LOCAL_WINDOW_SIZE]
        random_window = random_seq[start:start + LOCAL_WINDOW_SIZE]
        true_mfe = calculate_mfe(window)
        random_mfe = calculate_mfe(random_window)
        mfe_features[f'local_mfe_{start}'] = true_mfe
        mfe_features[f'local_delta_mfe_{start}'] = true_mfe - random_mfe

    local_mfe_vals = [
        mfe_features[f'local_mfe_{start}']
        for start in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE)
    ]
    local_delta_vals = [
        mfe_features[f'local_delta_mfe_{start}']
        for start in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE)
    ]
    
    mfe_features['average_local_mfe'] = fmean(local_mfe_vals) if local_mfe_vals else 0
    mfe_features['average_local_delta_mfe'] = fmean(local_delta_vals) if local_delta_vals else 0
    
    return mfe_features

def calculate_LFE_features(features):
    """
    Calculate local folding energy (LFE) features for each gene in the DataFrame.
    
    Parameters
    ----------
    features : pandas.DataFrame
        DataFrame with an "ORF" column containing nucleotide sequences.
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with appended LFE features.
    """
    print("Calculating local folding energy features...")
    lfe_data = features['ORF'].apply(calculate_local_mfe_features)
    lfe_df = pd.DataFrame(lfe_data.tolist()).fillna(0)
    return pd.concat([features, lfe_df.set_index(features.index)], axis=1)


