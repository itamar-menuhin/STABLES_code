from Bio.Seq import Seq
import ViennaRNA as VRNA
import random
import pandas as pd
from statistics import fmean

# Constants
LOCAL_WINDOW_SIZE = 40
STEP_SIZE = 10


def calculate_mfe(sequence):
    """
    Calculate the Minimum Free Energy (MFE) of an RNA sequence.

    Parameters:
    sequence (str): The RNA sequence.

    Returns:
    float: The MFE value.
    """
    rna_sequence = str(Seq(sequence).transcribe())  # Transcribe DNA to RNA
    return VRNA.fold(rna_sequence)[1]


def generate_random_permuted_sequence(dna_sequence):
    """
    Generate a random permuted DNA sequence while preserving amino acid translation.

    Parameters:
    dna_sequence (str): The DNA sequence.

    Returns:
    str: The permuted DNA sequence.
    """
    codons = [dna_sequence[i:i + 3] for i in range(0, len(dna_sequence), 3)]
    random.shuffle(codons)
    return ''.join(codons)


def calculate_local_mfe_features(dna_sequence):
    """
    Calculate local MFE and delta MFE features for a DNA sequence.

    Parameters:
    dna_sequence (str): The DNA sequence.

    Returns:
    dict: A dictionary containing local MFE and delta MFE features.
    """
    random_sequence = generate_random_permuted_sequence(dna_sequence)
    mfe_features = {}

    # Calculate local MFE and delta MFE
    for start_index in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE):
        window = dna_sequence[start_index:start_index + LOCAL_WINDOW_SIZE]
        random_window = random_sequence[start_index:start_index + LOCAL_WINDOW_SIZE]

        true_mfe = calculate_mfe(window)
        random_mfe = calculate_mfe(random_window)

        mfe_features[f'local_mfe_{start_index}'] = true_mfe
        mfe_features[f'local_delta_mfe_{start_index}'] = true_mfe - random_mfe

    # Calculate averages
    local_mfe_values = list(mfe_features[f'local_mfe_{start_index}'] for start_index in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE))
    local_delta_mfe_values = list(mfe_features[f'local_delta_mfe_{start_index}'] for start_index in range(0, len(dna_sequence) - LOCAL_WINDOW_SIZE + 1, STEP_SIZE))

    mfe_features['average_local_mfe'] = fmean(local_mfe_values) if local_mfe_values else 0
    mfe_features['average_local_delta_mfe'] = fmean(local_delta_mfe_values) if local_delta_mfe_values else 0

    return mfe_features


def calculate_LFE_features(features):
    """
    Calculate local folding energy (LFE) features for each gene.

    Parameters:
    features (pandas.DataFrame): The gene data.

    Returns:
    pandas.DataFrame: The updated gene data with LFE features.
    """
    print("Calculating local folding energy features...")
    lfe_data = features['ORF'].apply(calculate_local_mfe_features)
    return pd.concat([features, pd.DataFrame(lfe_data.tolist(), index=features.index)], axis=1)



