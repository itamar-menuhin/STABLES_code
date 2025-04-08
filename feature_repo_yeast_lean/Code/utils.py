#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - Codon usage bias and sequence utility functions

This module provides utility functions for calculating various measures of codon usage bias,
sequence context analysis, and sequence processing for gene expression prediction.

Key features include:
- Codon Adaptation Index (CAI) calculation
- Relative Codon Adaptation (RCA) weights computation
- tRNA Adaptation Index (tAI) integration
- Start codon context analysis via Position-Specific Scoring Matrix (PSSM)
- Sliding window analysis for local sequence features

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Sharp PM, Li WH. The codon Adaptation Index--a measure of directional synonymous
        codon usage bias, and its potential applications. Nucleic Acids Res. 1987;15(3):1281-95.
    [2] dos Reis M, Savva R, Wernisch L. Solving the riddle of codon usage preferences:
        a test for translational selection. Nucleic Acids Res. 2004;32(17):5036-44.
    [3] Kozak M. Regulation of translation via mRNA structure in prokaryotes and eukaryotes.
        Gene. 1999;234(2):187-208.
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqUtils import CodonAdaptationIndex
from statistics import geometric_mean
from Code.generate_highly_expressed_genes import load_highly_expressed_genes

# Constants - DO NOT CHANGE - preserves output alignment
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data')


def calculate_CAI_and_RCA_weights():
    """
    Calculate Codon Adaptation Index (CAI) and Relative Codon Adaptation (RCA) weights.
    
    This function computes two measures of codon usage bias:
    1. CAI weights: Based on relative synonymous codon usage in reference gene sets
    2. RCA weights: Based on observed vs. expected codon frequencies considering
       nucleotide composition
       
    The weights are calculated for all genes ('all') and highly expressed genes ('he')
    separately to enable different reference sets for codon optimization.
    
    Returns
    -------
    None
        Results are saved to CAI.pkl and RCA.pkl in the data directory
        
    References
    ----------
    Sharp PM, Li WH. The codon Adaptation Index--a measure of directional synonymous
    codon usage bias, and its potential applications. Nucleic Acids Res. 1987;15(3):1281-95.
    """
    gene_data = pd.read_csv(os.path.join(DATA_DIR, 'genes.csv'))
    gene_data = gene_data[gene_data['ORF'].str.len() % 3 == 0].set_index('gene')['ORF'].to_dict()

    CAI_weights, RCA_weights = {}, {}
    cai_calculator = CodonAdaptationIndex()

    for reference in ['all', 'he']:
        # CRITICAL: Use exact same reference set selection as original
        sequences = list(gene_data.values()) if reference == 'all' else load_highly_expressed_genes()
        concatenated = ''.join(sequences)
        codons = re.findall('.{3}', concatenated)

        # Calculate CAI weights using BioPython's CodonAdaptationIndex
        cai_calculator.generate_index(codons)
        CAI_weights[reference] = cai_calculator.index

        # Calculate RCA weights (Relative Codon Adaptation)
        # CRITICAL: Use exact same calculation method as original
        codon_frequencies = pd.Series(codons).value_counts(normalize=True)
        nucleotide_distribution = {
            pos: pd.Series(codons).str[pos].value_counts(normalize=True).to_dict() for pos in range(3)
        }
        RCA_weights[reference] = {
            codon: codon_frequencies[codon] / np.prod([nucleotide_distribution[pos].get(codon[pos], 0.5) for pos in range(3)])
            for codon in codon_frequencies.index
        }

    # Save results to pickle files
    pickle.dump(CAI_weights, open(os.path.join(DATA_DIR, 'CAI.pkl'), 'wb'))
    pickle.dump(RCA_weights, open(os.path.join(DATA_DIR, 'RCA.pkl'), 'wb'))


def combine_CUB_weights():
    """
    Combine Codon Usage Bias (CUB) weights from multiple metrics.
    
    This function integrates three different measures of codon usage bias:
    1. CAI (Codon Adaptation Index): For both all genes and highly expressed genes
    2. RCA (Relative Codon Adaptation): For both all genes and highly expressed genes
    3. tAI (tRNA Adaptation Index): Optimized for S. cerevisiae
    
    The combined weights provide a comprehensive set of metrics for
    analyzing translational efficiency of coding sequences.
    
    Returns
    -------
    None
        Results are saved to CUB_weights.pkl in the data directory
        
    References
    ----------
    dos Reis M, Savva R, Wernisch L. Solving the riddle of codon usage preferences:
    a test for translational selection. Nucleic Acids Res. 2004;32(17):5036-44.
    """
    # CRITICAL: Use exact same file paths as original
    CAI_weights = pickle.load(open(os.path.join(DATA_DIR, 'CAI.pkl'), 'rb'))
    RCA_weights = pickle.load(open(os.path.join(DATA_DIR, 'RCA.pkl'), 'rb'))
    tAI_weights = pd.read_excel(os.path.join(DATA_DIR, 'tAI.xls'), sheet_name='tAI', header=5).set_index('Codon')['S. Cerevisiae'].to_dict()

    # CRITICAL: Use exact same dictionary keys as original
    combined_weights = {
        "CAI_all": CAI_weights["all"],  # CAI weights for all genes
        "CAI_he": CAI_weights["he"],    # CAI weights for highly expressed genes
        "RCA_all": RCA_weights["all"],  # RCA weights for all genes
        "RCA_he": RCA_weights["he"],    # RCA weights for highly expressed genes
        "tAI": tAI_weights,            # tRNA Adaptation Index weights
    }
    pickle.dump(combined_weights, open(os.path.join(DATA_DIR, "CUB_weights.pkl"), "wb"))


def calculate_ATG_PSSM():
    """
    Calculate Position-Specific Scoring Matrix (PSSM) for ATG start codon context.
    
    This function analyzes the nucleotide frequencies in positions following the
    start codon in highly expressed genes. The resulting PSSM can be used to
    score the translation initiation context of genes, which is known to
    significantly influence translation efficiency.
    
    Returns
    -------
    dict
        A dictionary mapping position indices to nucleotide probability distributions
        
    References
    ----------
    Kozak M. Regulation of translation via mRNA structure in prokaryotes and eukaryotes.
    Gene. 1999;234(2):187-208.
    """
    # CRITICAL: Use exact same reference set as original
    highly_expressed_genes = load_highly_expressed_genes()
    pssm = {position: {} for position in range(3)}

    # Count nucleotide occurrences at each position after ATG
    # CRITICAL: Use the same position indexing as original
    for gene in highly_expressed_genes:
        for position in range(3):
            nucleotide = gene[position + 3] if len(gene) > position + 3 else None
            if nucleotide:
                pssm[position][nucleotide] = pssm[position].get(nucleotide, 0) + 1

    # Normalize counts to frequencies
    # CRITICAL: Use exact same normalization method as original
    for position, counts in pssm.items():
        total = sum(counts.values())
        pssm[position] = {nucleotide: count / total for nucleotide, count in counts.items()}

    return pssm


def add_AA_seq():
    """
    Add amino acid sequences to the genes dataset.
    
    This function translates DNA sequences to amino acid sequences and
    adds them as a new column in the genes.csv file. It enables protein-level
    feature analysis such as protein disorder prediction and chemical properties.
    
    Returns
    -------
    None
        The genes.csv file is updated with a new 'AA' column
    """
    genes = pd.read_csv(os.path.join(DATA_DIR, 'genes.csv'))
    
    # CRITICAL: Use exact same translation method as original
    genes['AA'] = genes['ORF'].apply(lambda orf: str(Seq(orf).translate(to_stop=False)))
    genes.to_csv(os.path.join(DATA_DIR, 'genes.csv'), index=False)


def calculate_sliding_window_features(sequence, num_windows, window_length, slide_step, feature_function=None, codon_usage_bias_weights=None):
    """
    Calculate features over sliding windows along a sequence.
    
    This function divides a sequence into overlapping windows and calculates
    features for each window. It's commonly used to analyze local patterns in
    codon usage, RNA folding energy, or other position-dependent properties.
    
    Parameters
    ----------
    sequence : str
        The nucleotide sequence to analyze
    num_windows : int
        Number of windows to analyze
    window_length : int
        Length of each window in nucleotides
    slide_step : int
        Step size for sliding the window
    feature_function : callable, optional
        Function to calculate features for each window
    codon_usage_bias_weights : dict, optional
        Dictionary of codon usage bias weights
        
    Returns
    -------
    list
        List of features for each window
    """
    # Default feature function calculates geometric mean of CUB weights
    # CRITICAL: Use exact same default function as original
    if feature_function is None:
        feature_function = lambda seq: geometric_mean([
            codon_usage_bias_weights.get(str(codon), 0)
            for codon in Seq(seq).translate(to_stop=False).split('*')  # Split by stop codons
            if codon_usage_bias_weights.get(str(codon), 0) > 0
        ])

    # Handle short sequences
    # CRITICAL: Use exact same handling of short sequences as original
    if len(sequence) < window_length:
        return [feature_function(sequence)] * num_windows

    # Calculate features for each window
    # CRITICAL: Use exact same window calculation as original
    return [
        feature_function(sequence[start:start + window_length])
        for start in range(0, len(sequence) - window_length + 1, slide_step)
    ]


def sliding_window(sequence, window_size, step_size):
    """
    Generate sliding windows from a sequence.
    
    This generator function divides a sequence into overlapping windows
    of fixed size, sliding by a specified step size. It is useful for
    analyzing local patterns in sequences without loading the entire
    sequence into memory at once.
    
    Parameters
    ----------
    sequence : str
        The input nucleotide sequence
    window_size : int
        The size of each window in nucleotides
    step_size : int
        The step size for sliding the window
        
    Yields
    ------
    str
        The next window in the sequence
    """
    # CRITICAL: Use exact same window generation as original
    for i in range(0, len(sequence) - window_size + 1, step_size):
        yield sequence[i:i + window_size]


if __name__ == "__main__":
    # Main execution for testing and initialization
    calculate_CAI_and_RCA_weights()
    combine_CUB_weights()
    PSSM = calculate_ATG_PSSM()
    print("ATG PSSM:", PSSM)
    add_AA_seq()
    print("Amino acid sequences added to genes.csv")