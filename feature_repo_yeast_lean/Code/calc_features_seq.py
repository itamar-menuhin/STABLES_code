#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_seq.py - Sequence-based feature calculation module

Calculates fundamental sequence-based features including nucleotide composition and
amino acid k-mer frequencies. These features can reflect codon bias, GC content preferences,
and amino acid motifs that influence protein expression and function.

Key features calculated include:
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
        Gene data containing an 'ORF' column with nucleotide sequences.
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with nucleotide fraction features.
    """
    print("Calculating nucleotide fractions...")
    for gene in features.index:
        seq = features.loc[gene, "ORF"].upper()
        total = len(seq)
        features.loc[gene, "frac_A"] = seq.count("A") / total
        features.loc[gene, "frac_T"] = seq.count("T") / total
        features.loc[gene, "frac_G"] = seq.count("G") / total
        features.loc[gene, "frac_C"] = seq.count("C") / total
    return features


def extract_kmers(sequence, k):
    """
    Extract k-mers of a specific length from a sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    k : int
        k-mer length.
        
    Returns
    -------
    Counter
        Counter object with k-mer counts.
    """
    return Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))


def calc_AA_kmers(features):
    """
    Calculate amino acid k-mer features (k = 3, 4, 5) for each gene.
    
    Parameters
    ----------
    features : pandas.DataFrame
        Gene data with amino acid sequences in the 'AA' column.
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with k-mer features.
    """
    print("Calculating amino acid k-mers...")
    for gene in features.index:
        aa_seq = features.loc[gene, "AA"]
        for k in range(3, 6):
            features.loc[gene, f"kmer_{k}"] = len(extract_kmers(aa_seq, k))
    return features


def calculate_seq_features(features):
    """
    Calculate sequence-based features (nucleotide fractions and amino acid k-mers)
    for each gene.
    
    Parameters
    ----------
    features : pandas.DataFrame
        Gene data with 'ORF' and 'AA' columns.
        
    Returns
    -------
    pandas.DataFrame
        Updated gene data with sequence-based features.
    """
    print("Calculating sequence-based features...")
    features = calc_nuc_fraction(features)
    features = calc_AA_kmers(features)
    return features
