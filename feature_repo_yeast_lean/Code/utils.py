import os
import re
import pickle
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqUtils import CodonAdaptationIndex
from statistics import geometric_mean
from generate_highly_expressed_genes import load_highly_expressed_genes

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data')

def calculate_CAI_and_RCA_weights():
    """
    Calculate Codon Adaptation Index (CAI) and Relative Codon Adaptation (RCA) weights.
    These weights measure the codon usage bias in different reference sets.
    
    The function processes all genes ('all') and highly expressed genes ('he') separately
    and saves the results to pickle files.
    
    Returns:
        None: Results are saved to CAI.pkl and RCA.pkl
    """
    gene_data = pd.read_csv(os.path.join(DATA_DIR, 'genes.csv'))
    gene_data = gene_data[gene_data['ORF'].str.len() % 3 == 0].set_index('gene')['ORF'].to_dict()

    CAI_weights, RCA_weights = {}, {}
    cai_calculator = CodonAdaptationIndex()

    for reference in ['all', 'he']:
        sequences = list(gene_data.values()) if reference == 'all' else load_highly_expressed_genes()
        concatenated = ''.join(sequences)
        codons = re.findall('.{3}', concatenated)

        # Calculate CAI weights using BioPython's CodonAdaptationIndex
        cai_calculator.generate_index(codons)
        CAI_weights[reference] = cai_calculator.index

        # Calculate RCA weights (Relative Codon Adaptation)
        codon_frequencies = pd.Series(codons).value_counts(normalize=True)
        nucleotide_distribution = {
            pos: pd.Series(codons).str[pos].value_counts(normalize=True).to_dict() for pos in range(3)
        }
        RCA_weights[reference] = {
            codon: codon_frequencies[codon] / np.prod([nucleotide_distribution[pos].get(codon[pos], 0.5) for pos in range(3)])
            for codon in codon_frequencies.index
        }

    # Save results to pickle files
    pickle.dump(CAI_weights, open(os.path.join(DATA_DIR, 'CAI.pkl'), 'wb'))
    pickle.dump(RCA_weights, open(os.path.join(DATA_DIR, 'RCA.pkl'), 'wb'))

def combine_CUB_weights():
    """
    Combine Codon Usage Bias (CUB) weights from CAI, RCA, and tAI sources.
    
    This function loads previously calculated weights for different measures of codon bias
    and combines them into a single dictionary for easier access.
    
    Returns:
        None: Results are saved to CUB_weights.pkl
    """
    CAI_weights = pickle.load(open(os.path.join(DATA_DIR, 'CAI.pkl'), 'rb'))
    RCA_weights = pickle.load(open(os.path.join(DATA_DIR, 'RCA.pkl'), 'rb'))
    tAI_weights = pd.read_excel(os.path.join(DATA_DIR, 'tAI.xls'), sheet_name='tAI', header=5).set_index('Codon')['S. Cerevisiae'].to_dict()

    combined_weights = {
        "CAI_all": CAI_weights["all"],  # CAI weights for all genes
        "CAI_he": CAI_weights["he"],    # CAI weights for highly expressed genes
        "RCA_all": RCA_weights["all"],  # RCA weights for all genes
        "RCA_he": RCA_weights["he"],    # RCA weights for highly expressed genes
        "tAI": tAI_weights,            # tRNA Adaptation Index weights
    }
    pickle.dump(combined_weights, open(os.path.join(DATA_DIR, "CUB_weights.pkl"), "wb"))

def calculate_ATG_PSSM():
    """
    Calculate Position-Specific Scoring Matrix (PSSM) for ATG start codon context.
    
    This function analyzes the nucleotide frequencies around ATG start codons
    in highly expressed genes to create a position-specific scoring matrix.
    
    Returns:
        dict: A dictionary mapping position indices to nucleotide probability distributions
    """
    highly_expressed_genes = load_highly_expressed_genes()
    pssm = {position: {} for position in range(3)}

    # Count nucleotide occurrences at each position after ATG
    for gene in highly_expressed_genes:
        for position in range(3):
            nucleotide = gene[position + 3] if len(gene) > position + 3 else None
            if nucleotide:
                pssm[position][nucleotide] = pssm[position].get(nucleotide, 0) + 1

    # Normalize counts to frequencies
    for position, counts in pssm.items():
        total = sum(counts.values())
        pssm[position] = {nucleotide: count / total for nucleotide, count in counts.items()}

    return pssm

def add_AA_seq():
    """
    Add amino acid sequences to the genes dataset and save the updated dataset.
    
    This function translates DNA sequences to amino acid sequences and
    adds them as a new column in the genes.csv file.
    """
    genes = pd.read_csv(os.path.join(DATA_DIR, 'genes.csv'))
    genes['AA'] = genes['ORF'].apply(lambda orf: str(Seq(orf).translate(to_stop=False)))
    genes.to_csv(os.path.join(DATA_DIR, 'genes.csv'), index=False)

def calculate_sliding_window_features(sequence, num_windows, window_length, slide_step, feature_function=None, codon_usage_bias_weights=None):
    """
    Calculate features over sliding windows along a sequence.
    
    Parameters:
        sequence (str): The sequence to analyze.
        num_windows (int): Number of windows.
        window_length (int): Length of each window.
        slide_step (int): Sliding step size.
        feature_function (callable): Function to calculate features for each window.
        codon_usage_bias_weights (dict): Codon Usage Bias weights.

    Returns:
        list: List of features for each window.
    """
    # Default feature function calculates geometric mean of CUB weights
    if feature_function is None:
        feature_function = lambda seq: geometric_mean([
            codon_usage_bias_weights.get(str(codon), 0)
            for codon in Seq(seq).translate(to_stop=False).split('*')  # Split by stop codons
            if codon_usage_bias_weights.get(str(codon), 0) > 0
        ])

    # Handle short sequences
    if len(sequence) < window_length:
        return [feature_function(sequence)] * num_windows

    # Calculate features for each window
    return [
        feature_function(sequence[start:start + window_length])
        for start in range(0, len(sequence) - window_length + 1, slide_step)
    ]

def sliding_window(sequence, window_size, step_size):
    """
    Generate sliding windows from a sequence.
    
    Parameters:
        sequence (str): The input sequence.
        window_size (int): The size of each window.
        step_size (int): The step size for sliding.

    Yields:
        str: The next window in the sequence.
    """
    for i in range(0, len(sequence) - window_size + 1, step_size):
        yield sequence[i:i + window_size]

if __name__ == "__main__":
    # Main execution for testing and initialization
    calculate_CAI_and_RCA_weights()
    combine_CUB_weights()
    PSSM = calculate_ATG_PSSM()
    print("ATG PSSM:", PSSM)
    add_AA_seq()
    print("Amino acid sequences added to genes.csv")