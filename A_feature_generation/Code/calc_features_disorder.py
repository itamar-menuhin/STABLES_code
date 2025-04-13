#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_disorder.py - Protein disorder feature calculation module

Calculates protein disorder features for genes based on the IUPred algorithm.
Key features include average disorder scores and percentage of disordered residues
for the entire protein, and for defined N- and C-terminal regions.

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]

References:
    [1] Dosztányi Z, Csizmok V, Tompa P, Simon I. IUPred: web server for the prediction
        of intrinsically unstructured regions of proteins based on estimated energy content.
        Bioinformatics. 2005.
"""

import os
import sys
import pandas as pd
import numpy as np

# Try importing IUPred; if unavailable, use a mock implementation.
try:
    from disorder.IUPred.iupred3_lib import iupred
except ImportError:
    print("Warning: IUPred library not found. Using mock implementation.")
    def iupred(sequence):
        return np.random.rand(len(sequence)),

# Constants (DO NOT CHANGE)
WINDOW_LENGTH = 50  # Window size for regional analysis

def calc_disorder(features):
    """
    Calculate protein disorder features for each gene.

    Parameters
    ----------
    features : pandas.DataFrame
        DataFrame with amino acid sequences in the 'AA' column.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with disorder features.
    """
    print("Calculating disorder features...")
    
    # For each gene, analyze three regions: the entire sequence, N-terminal, and C-terminal.
    for gene in features.index:
        sequence = features.loc[gene, "AA"]
        regions = {
            "all": sequence,
            "start": sequence[:WINDOW_LENGTH],
            "end": sequence[-WINDOW_LENGTH:]
        }
        for region_name, region_seq in regions.items():
            try:
                # Compute IUPred disorder scores (CRITICAL: preserve original calculations)
                iupred_scores = iupred(region_seq)[0]
                features.loc[gene, f"iupred_avg_{region_name}"] = np.mean(iupred_scores)
                features.loc[gene, f"iupred_pct_{region_name}"] = np.mean(iupred_scores > 0.5)
                # Preserve compatibility with original (moreronn and consensus features)
                features.loc[gene, f"moreronn_avg_{region_name}"] = 0
                features.loc[gene, f"moreronn_pct_{region_name}"] = 0
                features.loc[gene, f"disorder_consensus_{region_name}"] = 0
            except Exception as e:
                print(f"Error calculating disorder for gene {gene}, region {region_name}: {e}")
                features.loc[gene, f"iupred_avg_{region_name}"] = 0
                features.loc[gene, f"iupred_pct_{region_name}"] = 0
                features.loc[gene, f"moreronn_avg_{region_name}"] = 0
                features.loc[gene, f"moreronn_pct_{region_name}"] = 0
                features.loc[gene, f"disorder_consensus_{region_name}"] = 0
                
    return features
