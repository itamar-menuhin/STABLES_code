import pandas as pd
import re
import pickle
from collections import Counter, defaultdict
import numpy as np
import os

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
CODON_TABLE_PATH = os.path.join(DATA_DIR, 'codon_tables.pkl')
HIGHLY_EXPRESSED_GENES_PATH = os.path.join(DATA_DIR, 'highly_expressed.pkl')
GENES_FILE_PATH = os.path.join(DATA_DIR, 'genes.csv')

def calculate_enc(codons, amino_acid_to_codons):
    """
    Calculate the Effective Number of Codons (ENC) for a given sequence of codons.

    Parameters:
        codons (list): List of codons in the sequence.
        amino_acid_to_codons (dict): Mapping of amino acids to their codons.

    Returns:
        float: The ENC value for the sequence.
    """
    degeneracy_frequencies = defaultdict(list)

    # Calculate codon usage frequencies for each amino acid
    for amino_acid, codon_list in amino_acid_to_codons.items():
        codon_counts = Counter(codon for codon in codons if codon in codon_list).values()
        if codon_counts:
            frequencies = [(count / sum(codon_counts)) ** 2 for count in codon_counts]
            degeneracy_frequencies[len(codon_list)].append(sum(frequencies))

    # Compute ENC
    enc_value = 2 + sum(
        (weight / np.average(degeneracy_frequencies[level]) if degeneracy_frequencies[level] else weight)
        for level, weight in zip([2, 3, 4, 6], [9, 1, 5, 3])
    )
    return enc_value

def load_highly_expressed_genes(top_percentage=0.05):
    """
    Load or calculate the list of highly expressed genes based on ENC values.

    Parameters:
        top_percentage (float): The top percentage of genes to consider as highly expressed.

    Returns:
        list: List of highly expressed genes.
    """
    # Check if highly expressed genes are already cached
    if os.path.isfile(HIGHLY_EXPRESSED_GENES_PATH):
        with open(HIGHLY_EXPRESSED_GENES_PATH, 'rb') as file:
            return pickle.load(file)

    # Load codon table
    with open(CODON_TABLE_PATH, 'rb') as file:
        amino_acid_to_codons, _ = pickle.load(file)

    # Load gene sequences
    gene_data_frame = pd.read_csv(GENES_FILE_PATH, delimiter=',').set_index('gene')[['ORF']]

    # Calculate ENC for each gene
    gene_data_frame['ENC'] = gene_data_frame['ORF'].apply(
        lambda orf: calculate_enc(re.findall('.{3}', orf), amino_acid_to_codons)
    )

    # Select top highly expressed genes
    top_genes = gene_data_frame.nsmallest(round(top_percentage * len(gene_data_frame)), 'ENC')
    highly_expressed_genes = top_genes['ORF'].tolist()

    # Cache the results
    with open(HIGHLY_EXPRESSED_GENES_PATH, 'wb') as file:
        pickle.dump(highly_expressed_genes, file)

    return highly_expressed_genes

if __name__ == "__main__":
    print("Generating highly expressed genes...")
    highly_expressed_genes = load_highly_expressed_genes()
    print(f"Generated {len(highly_expressed_genes)} highly expressed genes.")