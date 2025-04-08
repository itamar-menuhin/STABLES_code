#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_seq.py - Sequence-based feature calculation module

This module calculates fundamental sequence-based features including
nucleotide composition and amino acid k-mer frequencies. These features
can reflect codon bias, GC content preferences, and amino acid motifs
that influence protein expression and function.

The key features calculated include:
- Nucleotide fractions (A, T, G, C content)
- Amino acid k-mer frequencies (k = 3, 4, 5)

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Sharp PM, Cowe E, Higgins DG, Shields DC, Wolfe KH, Wright F.
        Codon usage patterns in Escherichia coli, Bacillus subtilis,
        Saccharomyces cerevisiae, Schizosaccharomyces pombe, Drosophila
        melanogaster and Homo sapiens; a review of the considerable
        within-species diversity. Nucleic Acids Res. 1988.
"""

import pandas as pd
from Bio.Seq import Seq
from collections import Counter


def calc_nuc_fraction(features):
    """
    Calculate nucleotide fractions (A, T, G, C) for each gene.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with an 'ORF' column containing nucleotide sequences
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with nucleotide fraction features
    """
    print("Calculating nucleotide fractions...")
    
    for gene in features.index:
        seq = features.loc[gene, "ORF"].upper()
        total_length = len(seq)
        
        # CRITICAL: Calculate fractions exactly as in original
        features.loc[gene, "frac_A"] = seq.count("A") / total_length
        features.loc[gene, "frac_T"] = seq.count("T") / total_length
        features.loc[gene, "frac_G"] = seq.count("G") / total_length
        features.loc[gene, "frac_C"] = seq.count("C") / total_length
    
    return features


def extract_kmers(sequence, k):
    """
    Extract k-mers of a specific length from a sequence.
    
    Parameters
    ----------
    sequence : str
        The amino acid sequence
    k : int
        The k-mer length
        
    Returns
    -------
    Counter
        A counter object with k-mer counts
    """
    return Counter(str(sequence)[i:i+k] for i in range(len(sequence) - k + 1))


def calc_AA_kmers(features):
    """
    Calculate amino acid k-mer features (k = 3, 4, 5) for each gene.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with amino acid sequences in the 'AA' column
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with k-mer features
    """
    print("Calculating amino acid k-mers...")
    k_values = range(3, 6)  # k = 3, 4, 5
    
    for gene in features.index:
        aa_seq = features.loc[gene, "AA"]
        
        # CRITICAL: Calculate unique k-mer counts exactly as in original
        for k in k_values:
            features.loc[gene, f"kmer_{k}"] = len(extract_kmers(aa_seq, k))
    
    return features


def calculate_seq_features(features):
    """
    Calculate sequence-based features (nucleotide fractions and amino acid k-mers) for each gene.

    Parameters:
    features (pandas.DataFrame): The gene data with 'ORF' and 'AA' columns.

    Returns:
    pandas.DataFrame: The updated gene data with sequence-based features.
    """
    print("Calculating sequence-based features...")
    features = calc_nuc_fraction(features)
    features = calc_AA_kmers(features)
    return features
