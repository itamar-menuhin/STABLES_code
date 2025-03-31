import os
from generate_highly_expressed_genes import load_highly_expressed_genes
from utils import calculate_CAI_and_RCA_weights, combine_CUB_weights, calculate_ATG_PSSM
from calc_features import calculate_features

# Define constants
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "../Data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../Output")

def run_pipeline(target_gene, output_name="target_features", calculate_constants=True):
    """
    Run the entire feature calculation pipeline.

    Parameters:
    target_gene (str): The target gene sequence for feature calculation.
    output_name (str): The name of the output file for the calculated features.
    calculate_constants (bool): Whether to calculate constant features for all genes.

    Returns:
    None
    """
    print("Starting the feature calculation pipeline...")

    # Step 1: Generate highly expressed genes
    print("Step 1: Generating highly expressed genes...")
    load_highly_expressed_genes()
    print("Highly expressed genes generated.")

    # Step 2: Calculate CAI and RCA weights
    print("Step 2: Calculating CAI and RCA weights...")
    calculate_CAI_and_RCA_weights()
    print("CAI and RCA weights calculated.")

    # Step 3: Combine CUB weights
    print("Step 3: Combining CUB weights...")
    combine_CUB_weights()
    print("CUB weights combined.")

    # Step 4: Calculate ATG PSSM
    print("Step 4: Calculating ATG PSSM...")
    calculate_ATG_PSSM()
    print("ATG PSSM calculated.")

    # Step 5: Calculate features
    print("Step 5: Calculating features...")
    calculate_features(target_gene, output_name=output_name, calculate_constants=calculate_constants)
    print("Feature calculation completed.")

    print(f"Pipeline completed. Results saved to {OUTPUT_DIR}.")

if __name__ == "__main__":
    RFP = "atggtgagcaagggcgaggaggataacatggccatcatcaaggagttcatgcgcttcaaggtgcacatggagggctccgtgaacggccacgagttcgagatcgagggcgagggcgagggccgcccctacgagggcacccagaccgccaagctgaaggtgaccaagggtggccccctgcccttcgcctgggacatcctgtcccctcagttcatgtacggctccaaggcctacgtgaagcaccccgccgacatccccgactacttgaagctgtccttccccgagggcttcaagtgggagcgcgtgatgaacttcgaggacggcggcgtggtgaccgtgacccaggactcctccctgcaggacggcgagttcatctacaaggtgaagctgcgcggcaccaacttcccctccgacggccccgtaatgcagaagaagaccatgggctgggaggcctcctccgagcggatgtaccccgaggacggcgccctgaagggcgagatcaagcagaggctgaagctgaaggacggcggccactacgacgctgaggtcaagaccacctacaaggccaagaagcccgtgcagctgcccggcgcctacaacgtcaacatcaagttggacatcacctcccacaacgaggactacaccatcgtggaacagtacgaacgcgccgagggccgccactccaccggcggcatggacgagctgtacaag".upper()
    GFP = "atgtccaagggtgaagagctatttactggggttgtacccattttggtagaactggacggagatgtaaacggacataaattctctgttagaggtgagggcgaaggcgatgccaccaatggtaaattgactctgaagtttatatgcactacgggtaaattacctgttccttggccaaccctagtaacaactttgacatatggtgttcaatgtttctcaagatacccagaccatatgaaaaggcatgatttctttaaaagtgctatgccagaaggctacgtgcaagagagaactatctcctttaaggatgacggtacgtataaaacacgagcagaagtgaaattcgaaggggatacactagttaatcgcatcgaattaaagggtatagactttaaggaagatggtaatattctcggccataaacttgagtataatttcaactcgcataatgtgtacattacagctgacaaacaaaagaacggaattaaagcgaattttaaaatcaggcacaacgtcgaagatgggtctgttcaacttgccgatcattatcagcaaaacacccctattggtgatggtccagtcttgttacccgataatcactacttaagcacacagtctagattgtcaaaagatccgaatgaaaagcgtgatcacatggttttattggaatttgtcaccgctgcaggaataactcacggaatggacgagctttataagggatcc".upper()
    insulin = 'atgaaattgaaaactgttagatctgctgttttgtcttctttgtttgcttctcaagttttgggtcaaccaattgatgatactgaatctcaaactacttctgttaatttgatggctgatgatactgaatctgcttttgctactcaaactaattctggtggtttggatgttgttggtttgatttctatggctgaagaaggtgaaccaaaaaaaagatttgttaatcaacatttgtgtggttctcatttggttgaagctttgtatttggtttgtggtgaaagaggtttcttttacactccaaaggaatggaagggtatcgttgaacaatgttgtacttctatctgttctttgtaccaattggaaaattattgtaat'.upper()

    # Calculate features for the example gene
    calculate_features(insulin, output_name="insulin", calculate_constants=True)
