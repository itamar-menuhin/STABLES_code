from Bio.SeqUtils.ProtParam import ProteinAnalysis

def calculate_chemical_features(features):
    """
    Calculate chemical properties (e.g., aliphatic index, GRAVY score) for amino acid sequences.

    Parameters:
    features (pandas.DataFrame): The gene data with amino acid sequences in the 'AA' column.

    Returns:
    pandas.DataFrame: The updated gene data with chemical properties.
    """
    print("Calculating chemical properties...")
    features['aliphatic_index'] = features['AA'].apply(lambda seq: ProteinAnalysis(seq).aliphatic_index())
    features['gravy'] = features['AA'].apply(lambda seq: ProteinAnalysis(seq).gravy())
    return features
