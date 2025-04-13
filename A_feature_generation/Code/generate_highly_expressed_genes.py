#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_highly_expressed_genes.py - Highly expressed gene identification module

This module identifies highly expressed genes in the yeast genome by calculating
the Effective Number of Codons (ENC) for each gene. Genes with lower ENC values
exhibit stronger codon usage bias and are typically highly expressed.
These genes can be used as a reference set for further CAI calculations.

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

import os
import re
import pickle
import pandas as pd

# Import the reusable ENC function from utils.py
from Code.utils import calculate_enc

# Constants - DO NOT CHANGE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
CODON_TABLE_PATH = os.path.join(DATA_DIR, 'codon_tables.pkl')
HIGHLY_EXPRESSED_GENES_PATH = os.path.join(DATA_DIR, 'highly_expressed.pkl')
GENES_FILE_PATH = os.path.join(DATA_DIR, 'genes.csv')

def load_highly_expressed_genes(top_percentage=0.05):
    """
    Load or calculate the list of highly expressed gene sequences based on ENC values.
    
    Parameters
    ----------
    top_percentage : float, optional
        Fraction of genes to select as highly expressed (default: 0.05).
        
    Returns
    -------
    list of str
        List of gene sequences (ORFs) for the highly expressed genes.
    """
    if os.path.isfile(HIGHLY_EXPRESSED_GENES_PATH):
        with open(HIGHLY_EXPRESSED_GENES_PATH, 'rb') as file:
            return pickle.load(file)
    
    # Load codon table for ENC calculation
    with open(CODON_TABLE_PATH, 'rb') as file:
        amino_acid_to_codons, _ = pickle.load(file)
    
    # Load gene sequences
    gene_df = pd.read_csv(GENES_FILE_PATH, delimiter=',').set_index('gene')[['ORF']]
    
    # CRITICAL: Calculate ENC using the exact same method as in utils.py
    gene_df['ENC'] = gene_df['ORF'].apply(
        lambda orf: calculate_enc(re.findall('.{3}', orf), amino_acid_to_codons)
    )
    
    # Select the top percentage of genes with the lowest ENC values
    top_genes = gene_df.nsmallest(round(top_percentage * len(gene_df)), 'ENC')
    highly_expressed_genes = top_genes['ORF'].tolist()
    
    # Save for future use
    with open(HIGHLY_EXPRESSED_GENES_PATH, 'wb') as file:
        pickle.dump(highly_expressed_genes, file)
    
    return highly_expressed_genes


if __name__ == "__main__":
    print("Generating highly expressed genes...")
    genes = load_highly_expressed_genes()
    print(f"Generated {len(genes)} highly expressed genes.")