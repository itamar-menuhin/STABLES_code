#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fusion_translation_utils.py - Translation and Codon Helper Functions

This module provides helper functions for:
    • Translating DNA sequences to amino acid sequences.
    • Listing synonymous codons for a given codon.
    • Producing a reverse genetic code table mapping amino acids to their codons.

The functions here preserve the original output and are designed for use in 
fusion protein design pipelines.

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]

References:
    [1] Standard BioPython Documentation.
    [2] Relevant literature on codon usage and translation.
"""

from Bio.Seq import Seq
from Bio.Data.CodonTable import unambiguous_dna_by_id, unambiguous_dna_by_name


def gene_seq_2_aa(gene_seq):
    """
    Translate a DNA sequence to its corresponding amino acid sequence.
    
    Parameters
    ----------
    gene_seq : str
        The DNA sequence to be translated.
        
    Returns
    -------
    str
        The translated amino acid sequence.
    
    Notes
    -----
    This function uses BioPython's Seq object for translation and assumes
    the standard genetic code with the first base as the start of the first codon.
    """
    gene_seq_aa = Seq(gene_seq)
    gene_seq_aa = gene_seq_aa.translate()
    return str(gene_seq_aa)


def altcodons(codon, table):
    """
    List codons that code for the same amino acid or are also stop codons.
    
    Parameters
    ----------
    codon : str
        The codon for which to list synonyms.
    table : int
        Genetic code table ID (1 is standard).
        
    Returns
    -------
    list
        List of synonymous codons.
        
    Notes
    -----
    For stop codons the function returns all stop codons.
    For regular codons it returns all codons that encode the same amino acid.
    """
    tab = unambiguous_dna_by_id[table]
    if codon in tab.stop_codons:
        return tab.stop_codons
    try:
        aa = tab.forward_table[codon]
    except:
        return []
    return [k for (k, v) in tab.forward_table.items() if v == aa]


def define_table():
    """
    Define a mapping of amino acids to their synonymous codons.
    
    Returns
    -------
    dict
        Dictionary where keys are amino acids and values are lists of corresponding codons.
        
    Notes
    -----
    This function creates a reverse genetic code table that maps each amino acid to all
    codons that encode it. Stop codons are excluded.
    """
    standard_table = unambiguous_dna_by_name["Standard"]
    reverse_genetic_code = standard_table.back_table.copy()
    for aa in reverse_genetic_code.keys():
        primary_codon = reverse_genetic_code[aa]
        syno_codons = altcodons(primary_codon, 1)
        reverse_genetic_code[aa] = syno_codons
    reverse_genetic_code.pop(None, None)
    return reverse_genetic_code





