import pandas as pd
import re
from statistics import fmean
import numpy as np

# Constants
CONTEXT_WINDOW = 150
ATG_WINDOW = 30  # Number of codons to consider for ATG-related features

def calculate_pssm_score(sequence, pssm_matrix):
    """
    Calculate the Position-Specific Scoring Matrix (PSSM) score for a given sequence.

    Parameters:
    sequence (str): The nucleotide sequence to score.
    pssm_matrix (pandas.DataFrame): The PSSM matrix.

    Returns:
    float: The calculated PSSM score.
    """
    return sum(pssm_matrix.loc[nucleotide, index] for index, nucleotide in enumerate(sequence))

def find_atg_locations(sequence, max_length=None):
    """
    Find ATG codon locations in a sequence.

    Parameters:
    sequence (str): The nucleotide sequence.
    max_length (int): Maximum length to consider in the sequence.

    Returns:
    list: List of ATG codon positions.
    """
    sequence = sequence[:max_length] if max_length else sequence
    return [match.start() for match in re.finditer('ATG', sequence) if match.start() % 3 == 0]

def calculate_atg_features(orf_sequence, utr_sequence, pssm_matrix):
    """
    Calculate ATG-related features for a given ORF and UTR sequence.

    Parameters:
    orf_sequence (str): The Open Reading Frame (ORF) sequence.
    utr_sequence (str): The Untranslated Region (UTR) sequence.
    pssm_matrix (pandas.DataFrame): The PSSM matrix.

    Returns:
    dict: A dictionary containing ATG-related features.
    """
    # Limit sequences to specific lengths
    orf_sequence = orf_sequence[:CONTEXT_WINDOW]
    utr_sequence = utr_sequence[-9:]

    # Find ATG locations
    utr_atg_locations = find_atg_locations(utr_sequence)
    orf_atg_locations = find_atg_locations(orf_sequence)

    # Exclude the main ATG from ORF locations
    if orf_atg_locations and orf_atg_locations[0] == 0:
        orf_atg_locations = orf_atg_locations[1:]

    # Calculate main ATG score
    main_atg_context = utr_sequence[-6:] + orf_sequence[:6]
    main_atg_score = calculate_pssm_score(main_atg_context, pssm_matrix)

    # Calculate ATG features within the first 30 codons of the ORF
    window_scores = [
        calculate_pssm_score(orf_sequence[loc - 6:loc + 6], pssm_matrix)
        for loc in orf_atg_locations if 6 <= loc <= ATG_WINDOW * 3
    ]

    return {
        "ATG_UTR_count": len(utr_atg_locations),
        "ATG_ORF_count": len(orf_atg_locations),
        "ATG_main_score": main_atg_score,
        "ATG_ORF_window_count": len(window_scores),
        "ATG_ORF_window_mean_absolute": fmean(window_scores) if window_scores else 0,
        "ATG_ORF_window_mean_relative": (fmean(window_scores) - main_atg_score) if window_scores else 0,
        "ATG_ORF_window_max_absolute": max(window_scores, default=0),
        "ATG_ORF_window_max_relative": (max(window_scores, default=0) - main_atg_score) if window_scores else 0,
    }

def calculate_ATG_features(features, pssm_matrix):
    """
    Calculate ATG-related features for each gene.

    Parameters:
    features (pandas.DataFrame): The gene data.
    pssm_matrix (pandas.DataFrame): The PSSM matrix.

    Returns:
    pandas.DataFrame: The updated gene data with ATG-related features.
    """
    print("Calculating ATG-related features...")
    
    atg_data = features.apply(
        lambda row: calculate_atg_features(row['ORF'], row['UTR'], pssm_matrix), axis=1
    )
    atg_df = pd.DataFrame(atg_data.tolist(), index=features.index)
    return pd.concat([features, atg_df], axis=1)

