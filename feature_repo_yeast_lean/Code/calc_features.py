import os
import pandas as pd
from os.path import join, isfile
from time import time
from Bio.Seq import Seq

# Import feature calculation functions
from calc_features_disorder import calculate_disorder_features
from calc_features_CUB import calculate_CUB_features
from calc_features_target import calculate_target_features
from calc_features_ATG import calculate_ATG_features
from calc_features_sORF import calculate_sORF_features
from calc_features_seq import calculate_nucleotide_fractions, calculate_AA_kmers
from calc_features_lfe import calculate_LFE_features
from calc_features_chemical import calculate_chemical_features
from calc_features_chimera import calculate_chimera_features

# Define directories
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = join(SCRIPT_DIR, '../raw_data')
OUTPUT_DIR = join(SCRIPT_DIR, '../Output')


def calculate_features(target_gene, output_name="", calculate_constants=False):
    """
    Calculate features for a given target gene and optionally calculate constant features.

    Parameters:
    target_gene (str): The target gene sequence.
    output_name (str): The name to use for the output file.
    calculate_constants (bool): Whether to calculate constant features for all genes.

    Returns:
    None
    """
    output_name = f"_{output_name}" if output_name else ""

    # Step 1: Calculate constant features for all genes
    if calculate_constants:
        features = load_gene_data()
        features = apply_feature_functions(features)
        save_final_features(features)

    # Step 2: Load precomputed features if constants are not recalculated
    else:
        features = load_precomputed_features()

    # Step 3: Calculate target gene-based features
    if target_gene:
        features = calculate_target_gene_features(features, target_gene, output_name)


def load_gene_data():
    """
    Load and preprocess gene data.

    Returns:
    pandas.DataFrame: The preprocessed gene data.
    """
    genes_file = join(DATA_DIR, 'genes.csv')
    features = pd.read_csv(genes_file, delimiter=',')
    features = features[features.ORF.apply(lambda x: (len(x) % 3) == 0)]
    features.set_index('gene', drop=False, inplace=True, verify_integrity=True)
    features.sort_index(inplace=True)
    return features


def apply_feature_functions(features):
    """
    Apply all feature calculation functions to the gene data.

    Parameters:
    features (pandas.DataFrame): The gene data.

    Returns:
    pandas.DataFrame: The updated gene data with calculated features.
    """
    feature_functions = [
        calculate_AA_features,
        calculate_chemical_features,
        calculate_disorder_features,
        calculate_nucleotide_fractions,
        calculate_AA_kmers,
        calculate_LFE_features,
        calculate_sORF_features,
        calculate_chimera_features,
        calculate_CUB_features,
        calculate_ATG_features
    ]

    for func_index, func in enumerate(feature_functions):
        feature_file = join(OUTPUT_DIR, f"gene_features_{func_index}.csv")
        if isfile(feature_file):
            features = pd.read_csv(feature_file, sep="\t", index_col=0)
        else:
            print(f"Calculating features using {func.__name__}...")
            start_time = time()
            print(features.columns)
            0/0
            features = func(features)
            end_time = time()
            print(f"Completed in {(end_time - start_time) / 60:.2f} minutes.")
            features.to_csv(feature_file, sep="\t")

    return features


def save_final_features(features):
    """
    Save the final combined features to a CSV file.

    Parameters:
    features (pandas.DataFrame): The gene data with all features.

    Returns:
    None
    """
    final_file = join(OUTPUT_DIR, 'gene_features.csv')
    features = features.fillna(0)
    if "gene.1" in features.columns:
        features = features.rename(columns={"gene.1": "gene"})
    features.to_csv(final_file, sep="\t")


def load_precomputed_features():
    """
    Load precomputed features from a CSV file.

    Returns:
    pandas.DataFrame: The precomputed features.
    """
    features_file = join(OUTPUT_DIR, 'gene_features.csv')
    features = pd.read_csv(features_file, sep="\t", index_col=0)
    if "gene.1" in features.columns:
        features = features.rename(columns={"gene.1": "gene"})
    return features


def calculate_target_gene_features(features, target_gene, output_name):
    """
    Calculate target gene-based features and save them to a file.

    Parameters:
    features (pandas.DataFrame): The gene data with precomputed features.
    target_gene (str): The target gene sequence.
    output_name (str): The name to use for the output file.

    Returns:
    pandas.DataFrame: The updated gene data with target gene-based features.
    """
    print("Calculating target gene-based features...")
    start_time = time()
    features = calculate_target_features(features, target_gene)
    end_time = time()
    print(f"Completed in {(end_time - start_time) / 60:.2f} minutes.")
    features = features.fillna(0)

    # Save the target gene-based features
    output_file = join(OUTPUT_DIR, f"{output_name}.csv")
    features.round(4).to_csv(output_file, sep="\t")
    return features


def calculate_AA_features(features):
    """
    Calculate amino acid sequences for all genes.

    Parameters:
    features (pandas.DataFrame): The gene data.

    Returns:
    pandas.DataFrame: The updated gene data with amino acid sequences.
    """
    print("Calculating amino acid sequences...")
    features['AA'] = features['ORF'].apply(lambda seq: str(Seq(seq).translate()))

    # Filter out invalid sequences
    print(f"Total genes: {features.shape[0]}")
    features = features[features['AA'].apply(lambda x: 'X' not in x)]
    print(f"Genes without 'X': {features.shape[0]}")
    features = features[features['AA'].apply(lambda x: '*' not in x[:-1])]
    print(f"Genes without extra stop codons: {features.shape[0]}")

    return features

