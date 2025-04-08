#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_highly_expressed_genes.py - Highly expressed gene identification module

This module identifies highly expressed genes in the yeast genome by calculating
the Effective Number of Codons (ENC) for each gene. Genes with lower ENC values
exhibit stronger codon usage bias, which is often correlated with higher
expression levels.

The key functions include:
- ENC calculation for measuring codon usage bias
- Selection of genes with the strongest codon usage bias
- Storage of the identified highly expressed genes for use in CAI calculations

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Wright F. The 'effective number of codons' used in a gene. Gene. 1990;87(1):23-9.
        doi:10.1016/0378-1119(90)90491-9
    [2] Sharp PM, Li WH. The codon Adaptation Index--a measure of directional synonymous
        codon usage bias, and its potential applications. Nucleic Acids Res. 1987;15(3):1281-95.
        doi:10.1093/nar/15.3.1281
"""

import pandas as pd
import re
import pickle
from collections import Counter, defaultdict
import numpy as np
import os

# Constants - DO NOT CHANGE - preserves output alignment and file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
CODON_TABLE_PATH = os.path.join(DATA_DIR, 'codon_tables.pkl')
HIGHLY_EXPRESSED_GENES_PATH = os.path.join(DATA_DIR, 'highly_expressed.pkl')
GENES_FILE_PATH = os.path.join(DATA_DIR, 'genes.csv')


def calculate_enc(codons, amino_acid_to_codons):
    """
    Calculate the Effective Number of Codons (ENC) for a given sequence of codons.
    
    ENC is a measure of codon usage bias that ranges from 20 (extreme bias) to 61 
    (no bias). Genes with stronger codon bias (lower ENC values) are typically
    highly expressed in the organism.
    
    Parameters
    ----------
    codons : list
        List of codons in the sequence.
    amino_acid_to_codons : dict
        Mapping of amino acids to their codons.
        
    Returns
    -------
    float
        The ENC value for the sequence.
        
    Notes
    -----
    The implementation follows Wright's original formula:
    ENC = 2 + 9/F_2 + 1/F_3 + 5/F_4 + 3/F_6
    where F_i is the average homozygosity for amino acids with i synonymous codons.
    """
    # CRITICAL: This exact calculation method must be preserved to maintain output consistency
    degeneracy_frequencies = defaultdict(list)

    for amino_acid, codon_list in amino_acid_to_codons.items():
        codon_counts = Counter(codon for codon in codons if codon in codon_list).values()
        if codon_counts:
            frequencies = [(count / sum(codon_counts)) ** 2 for count in codon_counts]
            degeneracy_frequencies[len(codon_list)].append(sum(frequencies))

    return 2 + sum(
        (weight / np.average(degeneracy_frequencies[level]) if degeneracy_frequencies[level] else weight)
        for level, weight in zip([2, 3, 4, 6], [9, 1, 5, 3])
    )


def load_highly_expressed_genes(top_percentage=0.05):
    """
    Load or calculate the list of highly expressed genes based on ENC values.
    
    This function identifies genes with strong codon usage bias (low ENC values),
    which are likely to be highly expressed in the organism. These genes can be
    used as a reference set for calculating the Codon Adaptation Index.
    
    Parameters
    ----------
    top_percentage : float, optional
        The top percentage of genes to consider as highly expressed.
        Default is 0.05 (5% of genes).
        
    Returns
    -------
    list
        List of highly expressed gene sequences.
        
    Notes
    -----
    The function first checks if a precomputed list exists. If not, it calculates
    ENC values for all genes and selects the top percentage with lowest ENC values.
    """
    if os.path.isfile(HIGHLY_EXPRESSED_GENES_PATH):
        with open(HIGHLY_EXPRESSED_GENES_PATH, 'rb') as file:
            return pickle.load(file)

    # Load codon table for ENC calculation
    with open(CODON_TABLE_PATH, 'rb') as file:
        amino_acid_to_codons, _ = pickle.load(file)

    # Load gene sequences
    gene_data_frame = pd.read_csv(GENES_FILE_PATH, delimiter=',').set_index('gene')[['ORF']]
    
    # CRITICAL: Calculate ENC for each gene using the exact same method
    gene_data_frame['ENC'] = gene_data_frame['ORF'].apply(
        lambda orf: calculate_enc(re.findall('.{3}', orf), amino_acid_to_codons)
    )

    # CRITICAL: Select the top percentage of genes with lowest ENC values
    top_genes = gene_data_frame.nsmallest(round(top_percentage * len(gene_data_frame)), 'ENC')
    highly_expressed_genes = top_genes['ORF'].tolist()

    # Save results for future use
    with open(HIGHLY_EXPRESSED_GENES_PATH, 'wb') as file:
        pickle.dump(highly_expressed_genes, file)

    return highly_expressed_genes


if __name__ == "__main__":
    print("Generating highly expressed genes...")
    highly_expressed_genes = load_highly_expressed_genes()
    print(f"Generated {len(highly_expressed_genes)} highly expressed genes.")