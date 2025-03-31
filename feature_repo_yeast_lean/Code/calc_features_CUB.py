import os
import pickle
from Bio.Seq import Seq
from Bio.SeqUtils import CodonAdaptationIndex
from statistics import geometric_mean
import pandas as pd
import numpy as np

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
CODON_TABLES_PATH = os.path.join(DATA_DIR, "codon_tables.pkl")
CUB_WEIGHTS_PATH = os.path.join(DATA_DIR, "CUB_weights.pkl")
FIRST_LAST = 50  # Number of first/last codons to analyze


def load_codon_tables_and_weights():
    """
    Load codon tables and CUB weights from pickle files.

    Returns:
    tuple: A tuple containing amino acid to codon mapping and CUB weights.
    """
    with open(CODON_TABLES_PATH, "rb") as handle:
        amino_acid_to_codon, _ = pickle.load(handle)

    with open(CUB_WEIGHTS_PATH, "rb") as handle:
        cub_weights = pickle.load(handle)

    return amino_acid_to_codon, cub_weights


def calculate_effective_number_of_codons(codons, amino_acid_to_codon):
    """
    Calculate the Effective Number of Codons (ENC) for a sequence.

    Parameters:
    codons (list): List of codons in the sequence.
    amino_acid_to_codon (dict): Mapping of amino acids to codons.

    Returns:
    float: The ENC value.
    """
    d_vec = {deg: [] for deg in [1, 2, 3, 4, 6]}

    for aa, codon_list in amino_acid_to_codon.items():
        codon_counts = [codons.count(codon) for codon in codon_list]
        total_counts = sum(codon_counts)
        if total_counts > 0:
            freqs = [(count / total_counts) ** 2 for count in codon_counts]
            d_vec[len(codon_list)].append(sum(freqs))

    for deg in d_vec:
        d_vec[deg] = np.mean(d_vec[deg]) if d_vec[deg] else 1 / deg

    return 2 + 9 / d_vec[2] + 1 / d_vec[3] + 5 / d_vec[4] + 3 / d_vec[6]


def calculate_RCBS(sequence):
    """
    Calculate Relative Codon Bias Score (RCBS) for a sequence.

    Parameters:
    sequence (str): The nucleotide sequence.

    Returns:
    float: The RCBS value.
    """
    codons = [sequence[i:i + 3] for i in range(0, len(sequence), 3)]
    nt_dist = {n: {} for n in range(3)}

    for n in range(3):
        for codon in codons:
            nt = codon[n]
            nt_dist[n][nt] = nt_dist[n].get(nt, 0) + 1

    for n in nt_dist:
        total = sum(nt_dist[n].values())
        for nt in nt_dist[n]:
            nt_dist[n][nt] /= total

    rcbs = [
        1 + (codons.count(codon) / len(codons) - np.prod([nt_dist[n][codon[n]] for n in range(3)])) /
        np.prod([nt_dist[n][codon[n]] for n in range(3)])
        for codon in codons
    ]

    return geometric_mean(rcbs) - 1


def calculate_codon_usage_bias(sequence, amino_acid_to_codon, cub_weights):
    """
    Calculate Codon Usage Bias (CUB) features for a sequence.

    Parameters:
    sequence (str): The nucleotide sequence.
    amino_acid_to_codon (dict): Mapping of amino acids to codons.
    cub_weights (dict): Codon usage bias weights.

    Returns:
    dict: A dictionary of CUB features.
    """
    seq_obj = Seq(sequence)
    codons = [str(seq_obj[i:i + 3]) for i in range(0, len(seq_obj), 3)]
    cub_features = {"ENC": calculate_effective_number_of_codons(codons, amino_acid_to_codon)}

    for score, weights in cub_weights.items():
        scores = [weights.get(codon, 0) for codon in codons if codon in weights]
        cub_features[score] = geometric_mean(scores) if scores else 0
        cub_features[f"{score}_std"] = np.std(scores) if scores else 0

    cub_features["RCBS"] = calculate_RCBS(sequence)
    cub_features[f"RCBS_f{FIRST_LAST}"] = calculate_RCBS(sequence[:FIRST_LAST * 3])
    cub_features[f"RCBS_l{FIRST_LAST}"] = calculate_RCBS(sequence[-FIRST_LAST * 3:])

    return cub_features


def calculate_CUB_features(features):
    """
    Calculate codon usage bias (CUB) features for each gene.

    Parameters:
    features (pandas.DataFrame): The gene data.

    Returns:
    pandas.DataFrame: The updated gene data with CUB features.
    """
    print("Calculating codon usage bias features...")
    amino_acid_to_codon, cub_weights = load_codon_tables_and_weights()
    cub_df = features['ORF'].apply(lambda seq: calculate_codon_usage_bias(seq, amino_acid_to_codon, cub_weights)).apply(pd.Series)
    return pd.concat([features, cub_df], axis=1)
