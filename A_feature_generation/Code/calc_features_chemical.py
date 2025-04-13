#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_chemical.py - Protein chemical properties feature calculation module

Calculates features related to the chemical properties of proteins, including:
  - GRAVY score: Grand average of hydropathy
  - Aliphatic index: Relative volume occupied by aliphatic side chains

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]

References:
    [1] Ikai A. Thermostability and aliphatic index of globular proteins. J Biochem. 1980.
    [2] Kyte J, Doolittle RF. A simple method for displaying the hydropathic character
        of a protein. J Mol Biol. 1982.
"""

from Bio.SeqUtils.ProtParam import ProteinAnalysis

def calc_chemical(features):
    """
    Calculate chemical properties for amino acid sequences.
    
    Parameters
    ----------
    features : pandas.DataFrame
        DataFrame containing amino acid sequences in the 'AA' column.
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with 'gravy' and 'aliphatic' features added.
    """
    print("Calculating chemical properties...")
    
    for gene in features.index:
        try:
            aa_sequence = features.loc[gene, "AA"].replace('*', '')
            # Skip invalid sequences: empty or contains non-standard amino acids
            if not aa_sequence or any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in aa_sequence):
                features.loc[gene, 'gravy'] = 0
                features.loc[gene, 'aliphatic'] = 0
                continue
            
            analysis = ProteinAnalysis(aa_sequence)
            features.loc[gene, 'gravy'] = analysis.gravy()
            features.loc[gene, 'aliphatic'] = analysis.aliphatic_index()
        except Exception as e:
            print(f"Error calculating chemical properties for gene {gene}: {e}")
            features.loc[gene, 'gravy'] = 0
            features.loc[gene, 'aliphatic'] = 0
    
    return features
