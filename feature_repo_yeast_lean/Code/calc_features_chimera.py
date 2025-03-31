import pandas as pd
import numpy as np
import os
from bisect import bisect_left

# Define constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
SUFFIX_ARRAY_PATH = os.path.join(DATA_DIR, "suffix_array.csv")

def score_current_sequence(sequence, suffix_list):
    """
    Calculate the score for the current sequence using binary search.

    Parameters:
    sequence (str): The sequence to score.
    suffix_list (list): List of suffixes from the suffix array.

    Returns:
    int: The score for the sequence.
    """
    score = 0
    for i in range(len(sequence)):
        prefix = sequence[:i + 1]
        pos = bisect_left(suffix_list, prefix)
        if pos < len(suffix_list) and suffix_list[pos].startswith(prefix):
            score += 1
        else:
            break
    return score

def score_gene(sequence, current_orf, suffix_array):
    """
    Calculate the chimera score for a gene.

    Parameters:
    sequence (str): The full sequence of the gene.
    current_orf (str): The ORF (Open Reading Frame) of the gene.
    suffix_array (pandas.DataFrame): The suffix array DataFrame.

    Returns:
    float: The chimera score for the gene.
    """
    suffix_list = suffix_array.loc[suffix_array["ORF"] != current_orf, "seq"].tolist()
    return np.mean([score_current_sequence(sequence[i:], suffix_list) for i in range(len(sequence))])

def calculate_chimera_features(features):
    """
    Add chimera features to the features DataFrame.

    Parameters:
    features (pandas.DataFrame): The input features DataFrame.

    Returns:
    pandas.DataFrame: The updated features DataFrame with chimera features.
    """
    if not os.path.isfile(SUFFIX_ARRAY_PATH):
        raise FileNotFoundError(f"Suffix array file not found: {SUFFIX_ARRAY_PATH}")

    suffix_array = pd.read_csv(SUFFIX_ARRAY_PATH)
    print("Calculating chimera features...")

    features["chimeraARS"] = features.apply(
        lambda row: score_gene(row["ORF"], row.name, suffix_array), axis=1
    )

    print("Chimera features calculation complete.")
    return features