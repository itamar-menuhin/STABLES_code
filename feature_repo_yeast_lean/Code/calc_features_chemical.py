#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_features_chemical.py - Protein chemical properties feature calculation module

This module calculates features related to the chemical properties of proteins,
which can significantly impact protein stability, solubility, and expression levels.

The key features calculated include:
- Aliphatic index: Relative volume occupied by aliphatic side chains
- GRAVY score: Grand average of hydropathy

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]

References:
    [1] Ikai A. Thermostability and aliphatic index of globular proteins.
        J Biochem. 1980.
    [2] Kyte J, Doolittle RF. A simple method for displaying the hydropathic
        character of a protein. J Mol Biol. 1982.
"""

from Bio.SeqUtils.ProtParam import ProteinAnalysis


def calc_chemical(features):
    """
    Calculate chemical properties for amino acid sequences.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The gene data with amino acid sequences in the 'AA' column
        
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with chemical property features
    """
    print("Calculating chemical properties...")
    
    for gene in features.index:
        try:
            # Get amino acid sequence, removing stop codons if present
            aa_sequence = features.loc[gene, "AA"].replace('*', '')
            
            # Skip empty or invalid sequences
            if not aa_sequence or any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in aa_sequence):
                features.loc[gene, 'gravy'] = 0
                features.loc[gene, 'aliphatic'] = 0
                continue
            
            # CRITICAL: Use exact same calculation as original
            analysis = ProteinAnalysis(aa_sequence)
            
            # CRITICAL: Use exact same feature names as original
            features.loc[gene, 'gravy'] = analysis.gravy()
            features.loc[gene, 'aliphatic'] = analysis.aliphatic_index()
            
        except Exception as e:
            print(f"Error calculating chemical properties for gene {gene}: {e}")
            features.loc[gene, 'gravy'] = 0
            features.loc[gene, 'aliphatic'] = 0
    
    return features
