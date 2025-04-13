#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fusion_codon_optimization.py - Codon Optimization Utilities

Provides functions to select optimal codons based on the tRNA Adaptation Index (tAI)
and to perform reverse translation of a protein sequence using these optimal codons.
Essential for the fusion protein design pipeline.

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]
"""

import pandas as pd
from Bio.Data.CodonTable import unambiguous_dna_by_id
from translation_utils import define_table


def tai_maximal_codon(candidate_codons, tAI_dict):
    """
    Return the codon with the highest tAI value from candidate_codons.
    """
    scores = [(tAI_dict[codon], codon) for codon in candidate_codons]
    return sorted(scores, reverse=True)[0][1]


def AA_to_codon_dict(tAI_csv_path):
    """
    Generate a dictionary that maps each amino acid to its optimal codon based on tAI scores.
    """
    df = pd.read_csv(tAI_csv_path, usecols=['Codon', 'wi'])
    genetic_code = unambiguous_dna_by_id[1].forward_table
    df['AA'] = df.Codon.map(genetic_code)
    df.to_csv(tAI_csv_path)  # Preserve behavior
    tAI_scores = df.set_index('Codon')['wi'].to_dict()
    genetic_code.update({'TGA': '*', 'TAA': '*', 'TAG': '*'})
    reverse_table = define_table()
    return {aa: tai_maximal_codon(reverse_table[aa], tAI_scores) for aa in reverse_table}


def AA2nt_translate(AA_seq, codon_dict):
    """
    Translate an amino acid sequence into a nucleotide sequence using the provided codon dictionary.
    """
    return ''.join(codon_dict[aa] for aa in AA_seq)