#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fusion_pipeline.py - Fusion Protein Design Pipeline

Orchestrates translation, linker selection, codon optimization, and sequence analysis
for designing fusion proteins.

Author: [Author Name]
Affiliation: [Institution]
Email: [Email]
Date: [Date]
License: [License Type]
References:
    [1] Chen X, et al. Adv Drug Deliv Rev. 2013.
    [2] Reddy Chichili VP, et al. Protein Sci. 2013.
"""

import time
import pandas as pd
from os import chdir

from fusion_translation_utils import gene_seq_2_aa
from fusion_linker_selection import best_fusion_linker
from fusion_codon_optimization import AA_to_codon_dict, AA2nt_translate
from fusion_sequence_analysis import find_suspect


def step_2(target_seq, host_seq, linkers_csv, tai_csv, optimize_target):
    """
    Execute translation, linker selection, codon optimization, and sequence analysis.
    """
    print("\n\nStarting step 2")
    start = time.time()
    target_aa = gene_seq_2_aa(target_seq).replace("*", "")
    host_aa = gene_seq_2_aa(host_seq).replace("*", "")
    df_linkers = best_fusion_linker(target_aa, host_aa, linkers_csv)
    codon_dict = AA_to_codon_dict(tai_csv)
    df_linkers.loc[:, 'linker_nt_sequence'] = df_linkers.linker_AA_sequence.apply(lambda x: AA2nt_translate(x, codon_dict))
    df_linkers.loc[:, 'suspects_linker'] = df_linkers.linker_nt_sequence.apply(lambda x: find_suspect(target_seq, x, host_seq, mode='linker'))
    new_target = target_seq
    if optimize_target:
        new_target = AA2nt_translate(target_aa, codon_dict)
    df_linkers.loc[:, 'target_nt_seq'] = new_target
    df_linkers.loc[:, 'host_nt_seq'] = host_seq
    print(f"Step 2 completed in {time.time() - start:.2f} seconds")
    return df_linkers, new_target


def single_calc(target, host_id, host_path, target_seq, linkers_csv, tai_csv, optimize_target=True):
    """
    Perform a complete calculation for a single target–host pair.
    """
    df_genome = pd.read_csv(host_path)
    if df_genome.shape[1] == 0:
        df_genome = pd.read_csv(host_path, delimiter='\t')
    host_seq = df_genome[df_genome.gene_id1 == host_id][['gene_ORF']].iloc[0, 0]
    df_linkers, new_target = step_2(target_seq, host_seq, linkers_csv, tai_csv, optimize_target)
    target_parts = target.split('_')
    df_linkers.loc[:, 'host'] = target_parts[0]
    df_linkers.loc[:, 'target_gene'] = target_parts[1]
    df_linkers.loc[:, 'suspects_target'] = df_linkers.linker_nt_sequence.apply(
        lambda x: find_suspect(new_target, x, host_seq, mode='target', max_L=15))
    df_linkers.loc[:, 'host_gene_id'] = host_id
    final = df_linkers[['host', 'target_gene', 'host_gene_id', 'linker_AA_sequence',
                        'suspects_linker', 'suspects_target', 'target_nt_seq',
                        'linker_nt_sequence', 'host_nt_seq']]
    print(final)
    return final


if __name__ == "__main__":
    chdir('/mnt/c/Users/itamar/Desktop/pythonProject_model_improvement')
    dict_target_gene = {
        'subtilis_TGFB1': 'ATGGGCGGAAAACATGATATATCCAGACGTCAATTTTTGAATTATACGCTCACAGGCGTAGGAGGTTTTATGGCGGCTAGTATGCTCATGCCTATGGTTCGCTTCGCACTCGATATGGCACTTGACACGAACTATTGTTTTAGCTCGACTGAGAAGAACTGTTGTGTACGTCAACTGTACATTGATTTCCGTAAAGACCTGGGGTGGAAGTGGATCCACGAGCCCAAGGGCTATCACGCCAACTTCTGTTTGGGACCATGCCCTTATATATGGTCTCTGGACACACAGTACAGCAAAGTTCTGGCACTGTACAATCAGCACAACCCGGGTGCATCTGCGGCTCCGTGTTGCGTCCCTCAGGCTTTAGAACCTTTACCAATTGTATATTATGTAGGACGAAAACCCAAAGTAGAGCAGCTTTCCAACATGATCGTTCGGTCATGTAAGTGTAGTTAG'.upper(),
        'kompas_FGF2': 'ATGAGATTTCCTTCAATTTTTACTGCTGTTTTATTCGCAGCATCCTCCGCATTAGCTGCTCCAGTCAACACTACAACAGAAGATGAAACGGCACAAATTCCGGCTGAAGCTGTCATCGGTTACTCAGATTTAGAAGGGGATTTCGATGTTGCTGTTTTGCCATTTTCCAACAGCACAAATAACGGGTTATTGTTTATAAATACTACTATTGCCAGCATTGCTGCTAAAGAAGAAGGGGTATCTCTCGAGAAAAGAGAGGCTGAAGCTATGGCTGCAGGTTCTATTACTACATTGCCATCTTTGCCAGAAGATGGTGGATCTGGTGCTTTTCCACCTGGTCATTTTAAAGATCCAAAAAGATTGTATTGTAAAAATGGTGGATTTTTCTTGAGAATTCATCCAGATGGTAGAGTTGATGGTGTTAGAGAAAAATCTGATCCACATATTAAATTGCAATTGCAAGCTGAAGAGAGAGGTGTTGTCTCTATTAAAGGTGTTTGTGCTAATAGATATTTGGCTATGAAAGAAGATGGTAGATTGTTAGCTTCTAAATGTGTTACTGATGAATGTTTTTTCTTCGAAAGATTGGAATCTAATAACTATAATACTTATAGGTCTAGAAAATATTCTTCATGGTATGTTGCTTTGAAAAGAACTGGTCAATATAAATTGGGTCCAAAAACTGGTCCAGGTCAAAAAGCTATTTTGTTTTTGCCAATGTCTGCTAAATCTTGA'.upper()
    }
    dict_host_gene_id = {'ecoli_CDA2D': ['b0605', 'b4143', 'b1779']}
    list_targets = ['ecoli_CDA2D']
    path_linkers = 'In/utils/linkers.csv'
    output_path = 'data/predictions_linkers/linkers_ecoli_CDA2D.csv'
    
    ii = 0
    list_outputs = []
    try:
        list_outputs = [pd.read_csv(output_path)]
        ii = list_outputs[0].shape[0]
    except:
        pass
    list_outputs = []
    for target in list_targets:
        host = target.split('_')[0]
        genome_path = f'feature_repo_{host}/Data/escCol_genome.csv'
        tai_path = f'feature_repo_{host}/Data/tAI.csv'
        for gene_id in dict_host_gene_id[target]:
            if ii > 0:
                ii -= 1
                print(f'Already processed {target}_{gene_id}')
                continue
            curr = single_calc(target, gene_id, genome_path, dict_target_gene[target], path_linkers, tai_path)
            list_outputs.append(curr)
            pd.concat(list_outputs).to_csv(output_path)