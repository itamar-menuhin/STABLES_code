import pandas as pd
from Bio.Seq import Seq

# Define sORF window sizes in codons
SORF_WINDOWS = [30, 200]


def calculate_sORF(sequence):
    """
    Calculate small Open Reading Frame (sORF) features for a sequence.

    Parameters:
    sequence (str): The nucleotide sequence.

    Returns:
    dict: A dictionary of sORF features.
    """
    seq_obj = Seq(sequence)

    # Find all potential start and stop codons
    start_positions = [i for i in range(len(seq_obj)) if seq_obj[i:i + 3] == "ATG"]
    stop_positions = [
        i for i in range(len(seq_obj))
        if seq_obj[i:i + 3] in {"TAG", "TAA", "TGA"}
    ]

    # Calculate ORF lengths
    orf_lengths = [
        stop - start + 3
        for start in start_positions
        for stop in stop_positions
        if stop > start and (stop - start) % 3 == 0
    ]

    # Calculate global sORF features
    sORF_features = {
        "num_sORF": len(orf_lengths),
        "max_sORF_len": max(orf_lengths, default=0),
        "mean_sORF_len": sum(orf_lengths) / len(orf_lengths) if orf_lengths else 0,
    }

    # Calculate sORF features within defined windows
    for window in SORF_WINDOWS:
        window_orfs = [
            length for start, length in zip(start_positions, orf_lengths) if start <= window * 3
        ]
        sORF_features[f"num_sORF_win{window}"] = len(window_orfs)
        sORF_features[f"max_sORF_win{window}"] = max(window_orfs, default=0)
        sORF_features[f"mean_sORF_win{window}"] = sum(window_orfs) / len(window_orfs) if window_orfs else 0

    return sORF_features


def calculate_sORF_features(features):
    """
    Calculate small ORF (sORF) features for each gene.

    Parameters:
    features (pandas.DataFrame): The gene data with an 'ORF' column containing nucleotide sequences.

    Returns:
    pandas.DataFrame: The updated gene data with sORF features.
    """
    print("Calculating small ORF features...")
    # Apply `calculate_sORF` to each ORF and expand the resulting dictionaries into columns
    sORF_features_df = features["ORF"].apply(calculate_sORF).apply(pd.Series)

    # Merge the sORF features back into the original DataFrame
    return pd.concat([features, sORF_features_df], axis=1)
