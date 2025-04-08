#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py - Feature calculation pipeline orchestration module

This module orchestrates the complete pipeline for calculating genetic and proteomic
features for gene expression analysis. It integrates multiple pre-processing steps
and feature calculation modules to produce comprehensive feature sets for machine 
learning models that predict gene expression levels.

The pipeline consists of five sequential steps:
1. Generate highly expressed genes based on codon usage bias
2. Calculate Codon Adaptation Index (CAI) and Relative Codon Adaptation (RCA) weights
3. Combine codon usage bias (CUB) weights from different metrics
4. Calculate Position-Specific Scoring Matrix (PSSM) for ATG start codons
5. Calculate comprehensive feature set for target gene sequences

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Sharp PM, Li WH. The codon Adaptation Index--a measure of directional synonymous
        codon usage bias, and its potential applications. Nucleic Acids Res. 1987;15(3):1281-95.
    [2] Wright F. The 'effective number of codons' used in a gene. Gene. 1990;87(1):23-9.
    [3] Kozak M. Regulation of translation via mRNA structure in prokaryotes and eukaryotes.
        Gene. 1999;234(2):187-208.
"""

import os
from generate_highly_expressed_genes import load_highly_expressed_genes
from utils import calculate_CAI_and_RCA_weights, combine_CUB_weights, calculate_ATG_PSSM
from calc_features import calculate_features

# Define constants - DO NOT CHANGE - preserves output alignment
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "../Data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../Output")


def run_pipeline(target_gene, output_name="target_features", calculate_constants=True):
    """
    Run the entire feature calculation pipeline.

    The pipeline consists of five main steps:
    1. Generate highly expressed genes
    2. Calculate CAI and RCA weights
    3. Combine CUB weights
    4. Calculate ATG PSSM
    5. Calculate features for the target gene

    Parameters
    ----------
    target_gene : str
        The target gene sequence for feature calculation.
    output_name : str, optional
        The name of the output file for the calculated features.
        Default is "target_features".
    calculate_constants : bool, optional
        Whether to calculate constant features for all genes.
        Default is True.

    Returns
    -------
    None
        Results are saved to files in the output directory.
    """
    print("Starting the feature calculation pipeline...")

    # Step 1: Generate highly expressed genes
    print("Step 1: Generating highly expressed genes...")
    load_highly_expressed_genes()
    print("Highly expressed genes generated.")

    # Step 2: Calculate CAI and RCA weights
    print("Step 2: Calculating CAI and RCA weights...")
    calculate_CAI_and_RCA_weights()
    print("CAI and RCA weights calculated.")

    # Step 3: Combine CUB weights
    print("Step 3: Combining CUB weights...")
    combine_CUB_weights()
    print("CUB weights combined.")

    # Step 4: Calculate ATG PSSM
    print("Step 4: Calculating ATG PSSM...")
    calculate_ATG_PSSM()
    print("ATG PSSM calculated.")

    # Step 5: Calculate features
    print("Step 5: Calculating features...")
    calculate_features(target_gene, output_name=output_name, calculate_constants=calculate_constants)
    print("Feature calculation completed.")

    print(f"Pipeline completed. Results saved to {OUTPUT_DIR}.")

if __name__ == "__main__":
    # Example gene sequences
    RFP = "atggtgagcaagggcgaggaggataacatggccatcatcaaggagttcatgcgcttcaaggtgcacatggagggctccgtgaacggccacgagttcgagatcgagggcgagggcgagggccgcccctacgagggcacccagaccgccaagctgaaggtgaccaagggtggccccctgcccttcgcctgggacatcctgtcccctcagttcatgtacggctccaaggcctacgtgaagcaccccgccgacatccccgactacttgaagctgtccttccccgagggcttcaagtgggagcgcgtgatgaacttcgaggacggcggcgtggtgaccgtgacccaggactcctccctgcaggacggcgagttcatctacaaggtgaagctgcgcggcaccaacttcccctccgacggccccgtaatgcagaagaagaccatgggctgggaggcctcctccgagcggatgtaccccgaggacggcgccctgaagggcgagatcaagcagaggctgaagctgaaggacggcggccactacgacgctgaggtcaagaccacctacaaggccaagaagcccgtgcagctgcccggcgcctacaacgtcaacatcaagttggacatcacctcccacaacgaggactacaccatcgtggaacagtacgaacgcgccgagggccgccactccaccggcggcatggacgagctgtacaag".upper()
    GFP = "atgtccaagggtgaagagctatttactggggttgtacccattttggtagaactggacggagatmtaaacggacataaattctctgttagaggtgagggcgaaggcgatgccaccaatggtaaattgactctgaagtttatatgcactacgggtaaattacctgttccttggccaaccctagtaacaactttgacatatggtgttcaatgtttctcaagatacccagaccatatgaaaaggcatgatttctttaaaagtgctatgccagaaggctacgtgcaagagagaactatctcctttaaggatgacggtacgtataaaacacgagcagaagtgaaattcgaaggggatacactagttaatcgcatcgaattaaagggtatagactttaaggaagatggtaatattctcggccataaacttgagtataatttcaactcgcataatgtgtacattacagctgacaaacaaaagaacggaattaaagcgaattttaaaatcaggcacaacgtcgaagatgggtctgttcaacttgccgatcattatcagcaaaacacccctattggtgatggtccagtcttgttacccgataatcactacttaagcacacagtctagattgtcaaaagatccgaatgaaaagcgtgatcacatggttttattggaatttgtcaccgctgcaggaataactcacggaatggacgagctttataagggatcc".upper()
    insulin = 'atgaaattgaaaactgttagatctgctgttttgtcttctttgtttgcttctcaagttttgggtcaaccaattgatgatactgaatctcaaactacttctgttaatttgatggctgatgatactgaatctgcttttgctactcaaactaattctggtggtttggatgttgttggtttgatttctatggctgaagaaggtgaaccaaaaaaaagatttgttaatcaacatttgtgtggttctcatttggttgaagctttgtatttggtttgtggtgaaagaggtttcttttacactccaaaggaatggaagggtatcgttgaacaatgttgtacttctatctgttctttgtaccaattggaaaattattgtaat'.upper()

    # Calculate features for the insulin gene
    calculate_features(insulin, output_name="insulin", calculate_constants=True)
