import pandas as pd
from subprocess import check_output
from io import StringIO
from disorder.IUPred.iupred3_lib import iupred
import os

# Constants
WINDOW_LENGTH = 50
DISORDER_TOOL_PATH = os.path.join(os.path.dirname(__file__), "disorder", "MoreRONN")


def run_moreronn(sequence):
    """
    Run the MoreRONN tool on a given sequence and return the scores.

    Parameters:
    sequence (str): The amino acid sequence.

    Returns:
    pd.Series: A series of MoreRONN scores.
    """
    output = check_output(f"echo {sequence} | {DISORDER_TOOL_PATH} -s", shell=True)
    ronn_data = pd.read_csv(StringIO(output.decode('utf-8')), sep="\t", comment=">", header=None, names=["aa", "moreronn_score"]).dropna()
    return ronn_data["moreronn_score"]


def calculate_disorder(sequence):
    """
    Calculate disorder features for an amino acid sequence.

    Parameters:
    sequence (str): The amino acid sequence.

    Returns:
    dict: A dictionary of disorder features.
    """
    regions = {
        "all": sequence,
        "start": sequence[:WINDOW_LENGTH],
        "end": sequence[-WINDOW_LENGTH:]
    }

    disorder_features = {}
    for region, region_seq in regions.items():
        # Run MoreRONN and IUPred
        moreronn_scores = run_moreronn(region_seq)
        iupred_scores = iupred(region_seq)[0]

        # Calculate average, percentage, and consensus scores
        disorder_features.update({
            f"moreronn_avg_{region}": moreronn_scores.mean(),
            f"moreronn_pct_{region}": (moreronn_scores > 0.5).mean(),
            f"iupred_avg_{region}": iupred_scores.mean(),
            f"iupred_pct_{region}": (iupred_scores > 0.5).mean(),
            f"disorder_consensus_{region}": ((moreronn_scores > 0.5) & (iupred_scores > 0.5)).mean()
        })

    return disorder_features


def calculate_disorder_features(features):
    """
    Calculate protein disorder features for all genes in the dataset.

    Parameters:
    features (pd.DataFrame): The gene data with amino acid sequences.

    Returns:
    pd.DataFrame: The updated gene data with disorder features.
    """
    print("Calculating disorder features...")
    disorder_data = features['AA'].apply(calculate_disorder)
    return pd.concat([features, pd.DataFrame(disorder_data.tolist(), index=features.index)], axis=1)
