#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py - Feature calculation pipeline orchestration module

This module orchestrates the complete pipeline for calculating genetic and proteomic
features for gene expression analysis. It integrates multiple pre-processing steps
and feature calculation modules to produce comprehensive feature sets for machine
learning models that predict gene expression levels.

Pipeline steps:
    1. Generate highly expressed genes based on codon usage bias
    2. Calculate CAI and RCA weights
    3. Combine CUB weights from different metrics
    4. Calculate ATG PSSM for start codon context
    5. Calculate comprehensive feature set for target gene sequences

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]
References:
    [1] Sharp PM, Li WH. Nucleic Acids Res. 1987;15(3):1281-95.
    [2] Wright F. Gene. 1990;87(1):23-9.
    [3] Kozak M. Gene. 1999;234(2):187-208.
"""

import os
from generate_highly_expressed_genes import load_highly_expressed_genes
from utils import (calculate_CAI_and_RCA_weights, combine_CUB_weights, calculate_ATG_PSSM)
from calc_features import calculate_features

# Define constants (DO NOT CHANGE)
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "../Data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../Output")


def run_pipeline(target_gene, output_name="target_features", calculate_constants=True):
    """
    Run the complete feature calculation pipeline and save results.

    Parameters
    ----------
    target_gene : str
        The nucleotide sequence of the target gene.
    output_name : str, optional
        Output file name suffix (default "target_features").
    calculate_constants : bool, optional
        Whether to calculate constant features for all genes (default True).
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

    # Step 5: Calculate target gene features
    print("Step 5: Calculating target gene features...")
    calculate_features(target_gene, output_name=output_name, calculate_constants=calculate_constants)
    print("Feature calculation completed.")

    print(f"Pipeline completed. Results saved to {OUTPUT_DIR}.")


if __name__ == "__main__":
    # Example gene sequences
    RFP = ("ATGTGAGCAAGGGCGAGGAGGATAACATGGCCATCATCAAGGAGTTCATGCGCTTCAAGGTGCACATGGAGGGCTCCGTGAAC"
           "GGCCACGAGTTCGAGATCGAGGGCGAAGGGCGAAGGGCCGCCCCTACGAGGGCACCCAGACCGCCAAGCTGAAGGTGACCAAGGGT"
           "GGCCCCCTGCCCTT CGCCTGGGACATCCTGTCCCCTCAGTTCATGTACGGCTCCAAGGCCTACGTGAAGCACCCCGCCGACATCCCCGA"
           "CTACTTGAAGCTGTCC TTCCCCGAGGGCTTCAAGTGGGAGCGCGTGA TGA ACTT CGAGGACGGCGGCGT GGTGACC GTGACCCAG"
           "GACTCCTCCCTGCAGGACGGCGAGTTCATCTACAAGGTGAAGCTGCGCGG CACCAACTTCCCCTCCGACGGCCCCGTAATGCAGAAGAAGACCATGGG"
           "CTGGGAGGCCTCCTCCGAGCGGATGTACCCCGAGGACGGCGCCCTGAAGGGCGAGATCAAGCAGAGGCTGAAGCTGAAGGACGGCGGCC"
           "ACTACGACGCTGAGGTCAAGACCACCTACAAGGCCAAGAAGCCCGTG CAGCTGCCCGGCGCCTACAACGTC AACATCAAGTTGGACATCACCTC"
           "CCACAACGAGGACTACACCATCGTGGAACAGTACGAACGCGCCGAGGGCCGCCACTCCACC GGC GGCATGGACGAGCTGTACAAG").upper()
    GFP = ("ATGTCCAAGGGTGAAGAGCTATTTACTGGGGTTGTACCCATTTTGGTAGAACTGGACGGAGATMTA AACGGACATAAATTC"
           "TCTGTTAGAGGTGAGGGCGAAGGGC GATGCCACCAATGGTA AATTGACTCTGAAGTTTATATGCACTACGGGTAAATTACCTGTTCC"
           "TTGGCCAACCCTAGTAACAACTTTGACATATGGTGTTCAATGTTTCTCAAGATACCCAGACCATATGAAAA GG CATGATTTCTTTA AA"
           "AGTGCTATGCCAGAAGGCTACGTGCAAGAGAGAACTATCTCCTTTAAGGATGACGGTACGTATAAAACACGAGCAGAAGT GAAATT CGAAG"
           "GGGATACACTAGTTAA TCGCATCG AATTAAAGGGTATAGACTTT AAGGAAGATGGTAATATTCTCGGCCATAAACTTGAGTATAATTTCAAC"
           "TCGCATAATGTGTACATTACAGCTGACAAACAAAAG").upper()
    insulin = ("ATGAAATTGAAA ACTGTTAGATCTGCTGTTTTGTCTTCTT T GTTTGCTTCTCAAGTTTTGGGTCAACCAATTGATGATACTGAATCTCAAAC"
               "TACTTCTGTT AATTTGATGGCTGATGATACTGAATCTGCTTTTGCTACTCAA ACTAATTCTGGTGGT TTGGATGTTGTTGGTTT"
               "GATTTCTATGGCTGAAGAAGGTGAACC AAAAAAAGATTTGTT AATCAACATTTGTGTGGTTCTCATTTGGTTGAAGCTTTGTATTTGGTTT"
               "GTGGTGAAAGAGGTTTCTTTTACACTCCAAAGGAATGGAAGGGTATCGTTGAAC AATGTTGT ACTTCTATCTGTTCTTTGTACCAATTGGA"
               "AAATTATTGTAAT").upper()

    # Run pipeline for insulin gene as an example
    run_pipeline(insulin, output_name="insulin", calculate_constants=True)
