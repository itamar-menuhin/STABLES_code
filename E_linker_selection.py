#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fusion_linker_selection.py - Optimal Linker Selection for Fusion Proteins

Selects the best linker to connect two protein sequences by comparing the disorder profiles
of the target and host proteins before and after fusion.

Author: Itamar Menuhin-Gruman
Affiliation: Tel Aviv University
Email: imenuhin@gmail.com
Date: 13.4.2025
License: [License Type]
References:
    [1] Chen X, et al. Adv Drug Deliv Rev. 2013.
    [2] Reddy Chichili VP, et al. Protein Sci. 2013.
"""

import pandas as pd
from scipy.spatial import distance
from .In.utils import iupred2a_lib


def best_fusion_linker(target_seq, host_seq, linkers_csv):
    """
    Return a DataFrame with the optimal linker based on the minimal change in disorder scores.
    """
    df = pd.read_csv(linkers_csv)
    len_t, len_h = len(target_seq), len(host_seq)
    score_t_before = iupred2a_lib.iupred(target_seq)
    score_h_before = iupred2a_lib.iupred(host_seq)
    df.loc[:, 'distances_target'] = 0
    df.loc[:, 'distances_host'] = 0
    num = df.shape[0]
    for i, lid in enumerate(df.index):
        if i % 50 == 0:
            print(f'Completed {i} out of {num} linkers')
        linker = df.loc[lid, 'sequence']
        construct = target_seq + linker + host_seq
        scores = iupred2a_lib.iupred(construct)
        df.loc[lid, 'distances_target'] = distance.euclidean(scores[:len_t], score_t_before)
        df.loc[lid, 'distances_host'] = distance.euclidean(scores[-len_h:], score_h_before)
    df.loc[:, 'sum_distances'] = df['distances_target'] + df['distances_host']
    df.loc[:, 'score'] = 1 - (df['sum_distances'] - df['sum_distances'].min()) / (df['sum_distances'].max() - df['sum_distances'].min())
    best = df.sort_values('sum_distances').head(1)
    return best.rename(columns={'name': 'linker_name', 'score': 'linker_score', 'sequence': 'linker_AA_sequence'})[['linker_name', 'linker_score', 'linker_AA_sequence']]