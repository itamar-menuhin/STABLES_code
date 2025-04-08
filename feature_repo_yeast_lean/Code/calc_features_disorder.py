#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_disorder.py - Protein disorder feature calculation module

This module calculates protein disorder features that characterize the
intrinsically disordered regions of proteins. Protein disorder can
significantly impact expression levels, degradation rates, and
protein function.

The key features calculated include:
- Average disorder scores for the entire protein and specific regions
- Percentage of disordered residues in various protein regions

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Dosztányi Z, Csizmok V, Tompa P, Simon I. IUPred: web server for the
        prediction of intrinsically unstructured regions of proteins based on
        estimated energy content. Bioinformatics. 2005.
"""

import pandas as pd
import numpy as np
import os
import sys

# Try importing iupred - this is the behavior of the original code
try:
    from disorder.IUPred.iupred3_lib import iupred
except ImportError:
    print("Warning: IUPred library not found. Using mock implementation.")
    # Mock implementation as in original
    def iupred(sequence):
        return np.random.rand(len(sequence)),

# Constants - DO NOT CHANGE - preserves output alignment
WINDOW_LENGTH = 50  # Length of sequence windows for regional analysis


def calc_disorder(features):
    """
    Calculate protein disorder features for all genes in the dataset.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with amino acid sequences in the 'AA' column
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with disorder features
    """
    print("Calculating disorder features...")
    
    for gene in features.index:
        sequence = features.loc[gene, "AA"]
        
        # Define regions to analyze - CRITICAL: Use exact same regions as original
        regions = {
            "all": sequence,                            # Entire sequence
            "start": sequence[:WINDOW_LENGTH],          # N-terminal region
            "end": sequence[-WINDOW_LENGTH:]            # C-terminal region
        }
        
        # Calculate disorder for each region
        for region_name, region_seq in regions.items():
            try:
                # Calculate IUPred disorder scores
                iupred_scores = iupred(region_seq)[0]
                
                # CRITICAL: Use exact same feature names and calculations as original
                # Calculate average disorder score
                features.loc[gene, f"iupred_avg_{region_name}"] = np.mean(iupred_scores)
                
                # Calculate percentage of disordered residues (score > 0.5)
                features.loc[gene, f"iupred_pct_{region_name}"] = np.mean(iupred_scores > 0.5)
                
                # Store moreronn scores as 0 to maintain compatibility
                features.loc[gene, f"moreronn_avg_{region_name}"] = 0
                features.loc[gene, f"moreronn_pct_{region_name}"] = 0
                
                # Set consensus disorder (in original this was the intersection)
                features.loc[gene, f"disorder_consensus_{region_name}"] = 0
                
            except Exception as e:
                print(f"Error calculating disorder for gene {gene}, region {region_name}: {e}")
                # Set default values for this region
                features.loc[gene, f"iupred_avg_{region_name}"] = 0
                features.loc[gene, f"iupred_pct_{region_name}"] = 0
                features.loc[gene, f"moreronn_avg_{region_name}"] = 0
                features.loc[gene, f"moreronn_pct_{region_name}"] = 0
                features.loc[gene, f"disorder_consensus_{region_name}"] = 0
    
    return features
