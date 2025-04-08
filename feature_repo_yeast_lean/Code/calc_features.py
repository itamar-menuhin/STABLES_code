#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features.py - Main feature calculation orchestration module

This module serves as the central coordinator for calculating genetic and proteomic 
features for gene expression analysis. It integrates multiple specialized feature 
calculation modules to produce comprehensive feature sets for machine learning models
that predict gene expression levels.

The pipeline architecture:
1. Loads gene sequence data
2. Processes sequences to extract amino acid translations
3. Calculates features using specialized modules:
   - Codon usage bias features
   - Protein disorder features
   - Local RNA folding energy
   - Sequence chimerism
   - Chemical properties
   - Start codon context features
   - Shifted ORF features
   - And more
4. Saves computed features for model training and analysis

The modular design allows for easy extension with additional feature types
while maintaining a consistent interface for machine learning applications.

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Kudla G, Murray AW, Tollervey D, Plotkin JB. Coding-sequence
        determinants of gene expression in Escherichia coli. Science. 2009.
    [2] Tuller T, et al. An evolutionarily conserved mechanism for controlling
        the efficiency of protein translation. Cell. 2010.
"""

import os
import pandas as pd
from os.path import join, isfile
from time import time
from Bio.Seq import Seq
import sys

# Get absolute path to the parent directory - DO NOT CHANGE - preserves import structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import feature calculation functions - DO NOT CHANGE import order
from Code.calc_features_disorder import calc_disorder as calculate_disorder_features
from Code.calc_features_CUB import calc_CUB as calculate_CUB_features
from Code.calc_features_target import calculate_target_features
from Code.calc_features_ATG import calculate_ATG_features
from Code.calc_features_sORF import calculate_sORF_features
from Code.calc_features_seq import calc_nuc_fraction as calculate_nucleotide_fractions
from Code.calc_features_seq import calc_AA_kmers as calculate_AA_kmers
from Code.calc_features_lfe import calculate_LFE_features
from Code.calc_features_chemical import calc_chemical as calculate_chemical_features
from Code.calc_features_chimera import calculate_chimera_features

# Define directories
DATA_DIR = join(SCRIPT_DIR, '../Data')
OUTPUT_DIR = join(SCRIPT_DIR, '../Output')


def calculate_features(target_gene, output_name="", calculate_constants=False):
    """
    Calculate features for a given target gene and optionally calculate constant features.
    
    This is the main entry point for feature calculation that coordinates the entire process.
    It can either calculate all constant features for the gene set or use precomputed
    features and calculate only target-specific features.

    Parameters
    ----------
    target_gene : str
        The target gene sequence to calculate features for.
    output_name : str, optional
        The name to use for the output file.
    calculate_constants : bool, optional
        Whether to calculate constant features for all genes.

    Returns
    -------
    None
        Results are saved to files in the output directory.
    """
    output_name = f"_{output_name}" if output_name else ""

    if calculate_constants:
        # Calculate features for all genes in the dataset
        features = load_gene_data()
        features = apply_feature_functions(features)
        save_final_features(features)
    else:
        # Load precomputed features
        features = load_precomputed_features()

    if target_gene:
        # Calculate target-specific features
        features = calculate_target_gene_features(features, target_gene, output_name)


def load_gene_data():
    """
    Load and preprocess gene data from the genes.csv file.
    
    This function loads gene sequences from the data directory and performs
    initial validation checks to ensure the sequences are suitable for
    feature calculation (e.g., valid ORF lengths).
    
    Returns
    -------
    pandas.DataFrame
        The preprocessed gene data with gene as index.
    """
    genes_file = join(DATA_DIR, 'genes.csv')
    features = pd.read_csv(genes_file, delimiter=',')
    
    # Filter out genes with invalid ORF lengths (must be divisible by 3)
    features = features[features.ORF.apply(lambda x: len(x) % 3 == 0)]
    
    # Set gene name as index
    features.set_index('gene', inplace=True, verify_integrity=True)
    return features


def apply_feature_functions(features):
    """
    Apply all feature calculation functions to the gene data.
    
    This function orchestrates the application of all feature calculation
    modules to the gene dataset. The order of function application is important
    as some functions may depend on features calculated by previous functions.

    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with sequences.

    Returns
    -------
    pandas.DataFrame
        The updated gene data with calculated features.
    """
    # CRITICAL: Do not change the order of function application as it may affect
    # feature calculation dependencies and output consistency
    feature_functions = [
        calculate_AA_features,             # Amino acid sequence calculation
        calculate_chemical_features,       # Chemical properties
        calculate_disorder_features,       # Protein disorder features
        calculate_nucleotide_fractions,    # Nucleotide composition
        calculate_AA_kmers,                # Amino acid k-mers
        calculate_LFE_features,            # Local folding energy features
        calculate_sORF_features,           # Shifted ORF features
        calculate_chimera_features,        # Sequence chimerism features
        calculate_CUB_features,            # Codon usage bias features
        calculate_ATG_features             # Start codon context features
    ]

    # Apply each feature calculation function in sequence
    for func in feature_functions:
        print(f"Calculating features using {func.__name__}...")
        features = func(features)

    return features


def save_final_features(features):
    """
    Save the final combined features to a CSV file.

    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with all features.

    Returns
    -------
    None
        Saves features to a file and prints confirmation message.
    """
    final_file = join(OUTPUT_DIR, 'gene_features.csv')
    features.fillna(0).to_csv(final_file, sep="\t")
    print(f"Saved features to {final_file}")


def load_precomputed_features():
    """
    Load precomputed features from a CSV file.
    
    This function loads features that have been precomputed and saved to avoid
    redundant calculation of constant features.
    
    Returns
    -------
    pandas.DataFrame
        The precomputed features.
        
    Raises
    ------
    FileNotFoundError
        If the feature file is not found.
    """
    features_file = join(OUTPUT_DIR, 'gene_features.csv')
    if not isfile(features_file):
        raise FileNotFoundError(f"Feature file not found: {features_file}")
    return pd.read_csv(features_file, sep="\t", index_col=0)


def calculate_target_gene_features(features, target_gene, output_name):
    """
    Calculate target gene-based features and save them to a file.
    
    This function uses the precomputed features and the target gene sequence
    to calculate target-specific features like similarity metrics.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with precomputed features.
    target_gene : str
        The target gene sequence.
    output_name : str
        The name to use for the output file.

    Returns
    -------
    pandas.DataFrame
        The updated gene data with target gene-based features.
    """
    print("Calculating target gene-based features...")
    features = calculate_target_features(features, target_gene)
    output_file = join(OUTPUT_DIR, f"target_gene_features{output_name}.csv")
    features.fillna(0).round(4).to_csv(output_file, sep="\t")
    print(f"Target gene features saved to {output_file}")
    return features


def calculate_AA_features(features):
    """
    Calculate amino acid sequences for all genes.
    
    This function translates DNA sequences to amino acid sequences and
    filters out sequences with unexpected amino acids or premature stop codons.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with 'ORF' column containing DNA sequences.

    Returns
    -------
    pandas.DataFrame
        The updated gene data with 'AA' column containing amino acid sequences.
    """
    print("Calculating amino acid sequences...")
    
    # Convert DNA sequences to amino acid sequences
    features['AA'] = features['ORF'].apply(lambda seq: str(Seq(seq).translate()))
    
    # Filter out sequences with unexpected amino acids or premature stop codons
    features = features[features['AA'].apply(lambda x: 'X' not in x and '*' not in x[:-1])]
    
    return features
