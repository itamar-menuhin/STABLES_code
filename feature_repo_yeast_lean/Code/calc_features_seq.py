import pandas as pd
from Bio.Seq import Seq
from collections import Counter


def calculate_nucleotide_fractions(features):
    """
    Calculate nucleotide fractions (A, T, G, C) for each gene.

    Parameters:
    features (pandas.DataFrame): The gene data with an 'ORF' column containing nucleotide sequences.

    Returns:
    pandas.DataFrame: The updated gene data with nucleotide fractions.
    """
    print("Calculating nucleotide fractions...")
    nucleotide_df = features['ORF'].apply(lambda seq: Seq(seq).upper()).apply(lambda seq: {
        "frac_A": seq.count("A") / len(seq),
        "frac_C": seq.count("C") / len(seq),
        "frac_G": seq.count("G") / len(seq),
        "frac_T": seq.count("T") / len(seq),
    }).apply(pd.Series)

    return pd.concat([features, nucleotide_df], axis=1)


def calculate_AA_kmers(features):
    """
    Calculate amino acid k-mer features (k = 3, 4, 5) for each gene.

    Parameters:
    features (pandas.DataFrame): The gene data with amino acid sequences in the 'AA' column.

    Returns:
    pandas.DataFrame: The updated gene data with k-mer features.
    """
    print("Calculating amino acid k-mers...")
    k_lengths = range(3, 6)

    def extract_kmers(sequence, k):
        """
        Extract k-mers of a specific length from a sequence.

        Parameters:
        sequence (str): The amino acid sequence.
        k (int): The k-mer length.

        Returns:
        Counter: A counter object with k-mer counts.
        """
        return Counter(str(sequence)[i:i + k] for i in range(len(sequence) - k + 1))

    # Calculate k-mer counts for each sequence
    kmer_df = features['AA'].apply(lambda seq: {
        f"kmer_{k}": len(extract_kmers(Seq(seq), k)) for k in k_lengths
    }).apply(pd.Series)

    return pd.concat([features, kmer_df], axis=1)


def calculate_seq_features(features):
    """
    Calculate sequence-based features (nucleotide fractions and amino acid k-mers) for each gene.

    Parameters:
    features (pandas.DataFrame): The gene data with 'ORF' and 'AA' columns.

    Returns:
    pandas.DataFrame: The updated gene data with sequence-based features.
    """
    print("Calculating sequence-based features...")
    features = calculate_nucleotide_fractions(features)
    features = calculate_AA_kmers(features)
    return features
