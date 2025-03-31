import pickle
import re
from Bio.Seq import Seq
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp
from .utils import calc_ATG_PSSM
from .calc_features_CUB import calc_CUB
from .calc_features_sORF import calc_sORF
from .calc_features_seq import calc_nuc_fraction, calc_AA_kmers
from .calc_features_disorder import calc_disorder
from .calc_features_chemical import calc_chemical

# Constants
CODON_TABLE_PATH = '../Data/codon_tables.pkl'
DISTANCE_TYPES = ["L2", "L1", "spearman", "pearson", "KS"]

# Load codon table
with open(CODON_TABLE_PATH, 'rb') as handle:
    CODON_TABLE = pickle.load(handle)[0]


def calculate_target_features(features, target_sequence):
    """
    Calculate target-specific features for a given target gene.

    Parameters:
    features (pandas.DataFrame): The input features DataFrame.
    target_sequence (str): The target gene sequence.

    Returns:
    pandas.DataFrame: The updated features DataFrame with target-specific features.
    """
    print("Calculating target-specific features...")

    # Calculate target properties
    target_codon_freq, target_aa_freq = calculate_frequencies(target_sequence)

    # Precompute codon and amino acid frequencies for all sequences
    features[["codon_freq", "aa_freq"]] = features["ORF"].apply(calculate_frequencies).apply(pd.Series)

    # Calculate distances between target and endogenous properties
    for property_name, target_property in zip(["codon_freq", "aa_freq"], [target_codon_freq, target_aa_freq]):
        for distance_name in DISTANCE_TYPES:
            features[f"{property_name}_{distance_name}"] = features[property_name].apply(
                lambda endogenous_property: calculate_distance(target_property, endogenous_property, distance_name)
            )

    # Add features from other modules
    for func in [calc_CUB, calc_sORF, calc_nuc_fraction, calc_AA_kmers, calc_disorder, calc_chemical]:
        features = func(features)

    return features


def calculate_frequencies(sequence):
    """
    Calculate codon and amino acid frequencies for a sequence.

    Parameters:
    sequence (str): The nucleotide sequence.

    Returns:
    tuple: Codon frequencies and amino acid frequencies.
    """
    seq_obj = Seq(sequence)
    amino_acid_seq = seq_obj.translate(to_stop=True)

    codon_frequencies = [
        sequence.count(codon) / (len(sequence) // 3) for codon_list in CODON_TABLE.values() for codon in codon_list
    ]
    amino_acid_frequencies = [
        amino_acid_seq.count(aa) / len(amino_acid_seq) for aa in CODON_TABLE.keys()
    ]

    return codon_frequencies, amino_acid_frequencies


def calculate_distance(vector1, vector2, distance_type):
    """
    Calculate the distance between two vectors.

    Parameters:
    vector1 (list): The first vector.
    vector2 (list): The second vector.
    distance_type (str): The type of distance to calculate.

    Returns:
    float: The calculated distance.
    """
    if distance_type == "L2":
        return np.linalg.norm(np.array(vector1) - np.array(vector2), ord=2)
    elif distance_type == "L1":
        return np.linalg.norm(np.array(vector1) - np.array(vector2), ord=1)
    elif distance_type == "spearman":
        return spearmanr(vector1, vector2).correlation
    elif distance_type == "pearson":
        return pearsonr(vector1, vector2).statistic
    elif distance_type == "KS":
        return ks_2samp(vector1, vector2).statistic
    raise ValueError(f"Invalid distance type: {distance_type}")


def calculate_initiation_features(sequence):
    """
    Calculate initiation-related features for a sequence.

    Parameters:
    sequence (str): The nucleotide sequence.

    Returns:
    dict: A dictionary of initiation-related features.
    """
    sequence = sequence.upper()
    if not sequence.startswith("ATG"):
        raise ValueError("Sequence must start with ATG")

    window_size = 30  # In codons
    atg_positions = [match.start() for match in re.finditer("ATG", sequence) if match.start() % 3 == 0]
    filtered_positions = [pos for pos in atg_positions if (pos // 3) < window_size and (pos + 5) < len(sequence)]

    pssm_matrix = calc_ATG_PSSM()
    pssm_scores = [np.prod([pssm_matrix[i][sequence[pos + i]] for i in range(3)]) for pos in filtered_positions]

    return {
        "ATG_ORF": max(len(atg_positions) - 1, 0),
        f"ATG_ORF_window{window_size}": max(len(filtered_positions) - 1, 0),
        f"ATG_ORF_window{window_size}_mean": np.mean(pssm_scores) if pssm_scores else 0,
    }
