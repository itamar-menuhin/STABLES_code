#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fusion_sequence_analysis.py - Sequence Analysis for Fusion Protein Constructs

Detects potential issues (translational slippage, repetitive motifs, and recombination sites)
in fusion protein constructs.
  
Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]
"""

def find_slippage_1(target_seq, linker_seq, host_seq, mode):
    """
    Detect homopolymeric sequences causing translational slippage.
    """
    test_seq = target_seq[-3:] + linker_seq + host_seq[:3]
    if mode == 'target':
        test_seq = target_seq
    motifs = ['AAAA', 'CCCC', 'GGGG', 'TTTT']
    detected = [m for m in motifs if test_seq.find(m) > -1]
    return ','.join(detected)

def find_slippage_L(target_seq, linker_seq, host_seq, L, mode):
    """
    Detect repetitive motifs of length L that might cause translational slippage.
    """
    border = 3 * L - 1
    test_seq = target_seq[-border:] + linker_seq + host_seq[:border]
    if mode == 'target':
        test_seq = target_seq
    substrings = {test_seq[i:i+L] for i in range(len(test_seq)) if len(test_seq[i:i+L]) == L}
    motifs = sorted([s * 3 for s in substrings])
    detected = [m for m in motifs if test_seq.find(m) > -1]
    return ','.join(detected)

def find_recombination(target_seq, linker_seq, host_seq, L, mode):
    """
    Detect potential recombination sites based on motifs of length L.
    """
    border = L - 1
    test_seq = target_seq[-border:] + linker_seq + host_seq[:border]
    if mode == 'target':
        test_seq = target_seq
    substrings = {test_seq[i:i+L] for i in range(len(test_seq)) if len(test_seq[i:i+L]) == L}
    detected = []
    if mode == 'linker':
        for s in substrings:
            if target_seq.find(s) > -1 or host_seq.find(s) > -1:
                detected.append(s)
    else:
        for s in substrings:
            if linker_seq.find(s) > -1:
                detected.append(s)
                print(s, target_seq.find(s), linker_seq.find(s))
            if host_seq.find(s) > -1:
                detected.append(s)
                print(s, target_seq.find(s), host_seq.find(s))
    detected = sorted(set(detected))
    return ','.join(detected)

def find_suspect(target_seq, linker_seq, host_seq, mode='linker', max_L=12):
    """
    Combine outputs of slippage and recombination analysis into a summary string.
    """
    suspects = [find_slippage_1(target_seq, linker_seq, host_seq, mode)]
    for L in range(2, max_L):
        suspects.append(find_slippage_L(target_seq, linker_seq, host_seq, L, mode))
    suspects.append(find_recombination(target_seq, linker_seq, host_seq, max_L, mode))
    return ','.join(s for s in suspects if s)