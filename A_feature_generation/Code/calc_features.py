#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features.py - Main feature calculation orchestration module

This module serves as the central coordinator for calculating genetic and proteomic 
features for gene expression analysis. It integrates multiple specialized feature 
calculation modules to produce comprehensive feature sets for machine learning models
that predict gene expression levels.

Pipeline architecture:
1. Load gene sequence data
2. Process sequences to extract amino acid translations
3. Calculate features using specialized modules:
   - Codon usage bias features
   - Protein disorder features
   - Local RNA folding energy
   - Sequence chimerism
   - Chemical properties
   - Start codon context features
   - Shifted ORF features
   - And more
4. Save computed features for model training and analysis

Modular design allows easy extension with additional feature types while 
maintaining a consistent interface.

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]

References:
    [1] Kudla G, Murray AW, Tollervey D, Plotkin JB. Coding-sequence determinants of gene 
        expression in Escherichia coli. Science. 2009.
    [2] Tuller T, et al. An evolutionarily conserved mechanism for controlling the efficiency 
        of protein translation. Cell. 2010.
"""

import os
import sys
import pandas as pd
from os.path import join, isfile
from time import time
from Bio.Seq import Seq

# Set script directory and update sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import feature calculation functions - DO NOT CHANGE order
from Code.calc_features_disorder import calc_disorder as calculate_disorder_features
from Code.calc_features_CUB import calc_CUB as calculate_CUB_features
from Code.calc_features_target import calculate_target_features
from Code.calc_features_ATG import calculate_ATG_features
from Code.calc_features_sORF import calculate_sORF_features
from Code.calc_features_seq import calc_nuc_fraction as calculate_nucleotide_fractions, calc_AA_kmers as calculate_AA_kmers
from Code.calc_features_lfe import calculate_LFE_features
from Code.calc_features_chemical import calc_chemical as calculate_chemical_features
from Code.calc_features_chimera import calculate_chimera_features

# Define directories
DATA_DIR = join(SCRIPT_DIR, '../Data')
OUTPUT_DIR = join(SCRIPT_DIR, '../Output')


def calculate_features(target_gene, output_name="", calculate_constants=False):
    """
    Calculate features for a given target gene and optionally all constant features.

    Parameters
    ----------
    target_gene : str
        The target gene sequence for which features are calculated.
    output_name : str, optional
        Name used for the output file.
    calculate_constants : bool, optional
        Whether to compute constant features for all genes (True)
        or load precomputed features (False).

    Returns
    -------
    None
        Calculated features are saved to files in OUTPUT_DIR.
    """
    output_name = f"_{output_name}" if output_name else ""
    
    if calculate_constants:
        features = load_gene_data()
        features = apply_feature_functions(features)
        save_final_features(features)
    else:
        features = load_precomputed_features()

    if target_gene:
        features = calculate_target_gene_features(features, target_gene, output_name)


def load_gene_data():
    """
    Load and preprocess gene data from the genes.csv file.

    Returns
    -------
    pandas.DataFrame
        Preprocessed gene data with gene names as the index.
    """
    genes_file = join(DATA_DIR, 'genes.csv')
    features = pd.read_csv(genes_file, delimiter=',')
    features = features[features.ORF.apply(lambda x: len(x) % 3 == 0)]
    features.set_index('gene', inplace=True, verify_integrity=True)
    return features


def apply_feature_functions(features):
    """
    Apply all feature calculation functions sequentially.

    Parameters
    ----------
    features : pandas.DataFrame
        Gene data with sequence information.

    Returns
    -------
    pandas.DataFrame
        Updated gene data with added features.
    """
    # CRITICAL: Order of function application must not change
    feature_functions = [
        calculate_AA_features,             # Translate ORF to amino acids
        calculate_chemical_features,       # Chemical properties
        calculate_disorder_features,       # Protein disorder features
        calculate_nucleotide_fractions,    # Nucleotide composition
        calculate_AA_kmers,                # Amino acid k-mers
        calculate_LFE_features,            # Local RNA folding energy features
        calculate_sORF_features,           # Shifted ORF features
        calculate_chimera_features,        # Sequence chimerism features
        calculate_CUB_features,            # Codon usage bias features
        calculate_ATG_features             # Start codon context features
    ]
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
        Gene data with all calculated features.

    Returns
    -------
    None
        Features are saved to gene_features.csv in OUTPUT_DIR.
    """
    final_file = join(OUTPUT_DIR, 'gene_features.csv')
    features.fillna(0).to_csv(final_file, sep="\t")
    print(f"Saved features to {final_file}")


def load_precomputed_features():
    """
    Load precomputed features from the gene_features.csv file.

    Returns
    -------
    pandas.DataFrame
        The precomputed features.
        
    Raises
    ------
    FileNotFoundError
        If the feature file is unavailable.
    """
    features_file = join(OUTPUT_DIR, 'gene_features.csv')
    if not isfile(features_file):
        raise FileNotFoundError(f"Feature file not found: {features_file}")
    return pd.read_csv(features_file, sep="\t", index_col=0)


def calculate_target_gene_features(features, target_gene, output_name):
    """
    Calculate target gene-based features and save them to a file.

    Parameters
    ----------
    features : pandas.DataFrame
        Precomputed gene data.
    target_gene : str
        The target gene nucleotide sequence.
    output_name : str
        Suffix for the output filename.

    Returns
    -------
    pandas.DataFrame
        Updated gene data with target gene-specific features.
    """
    print("Calculating target gene-based features...")
    features = calculate_target_features(features, target_gene)
    output_file = join(OUTPUT_DIR, f"target_gene_features{output_name}.csv")
    features.fillna(0).round(4).to_csv(output_file, sep="\t")
    print(f"Target gene features saved to {output_file}")
    return features


def calculate_AA_features(features):
    """
    Translate DNA sequences (ORF) to amino acid sequences.

    Parameters
    ----------
    features : pandas.DataFrame
        Gene data with an 'ORF' column containing DNA sequences.

    Returns
    -------
    pandas.DataFrame
        Gene data with an added 'AA' column containing amino acid translations.
    """
    print("Calculating amino acid sequences...")
    features['AA'] = features['ORF'].apply(lambda seq: str(Seq(seq).translate()))
    features = features[features['AA'].apply(lambda x: 'X' not in x and '*' not in x[:-1])]
    return features
