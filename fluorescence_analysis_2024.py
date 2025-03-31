import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import scipy

def load_data(file_path):
    """
    Load the raw data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def plot_data(df_ground_truth, output_path):
    """
    Plot the data and save the figure.

    Parameters:
    df_ground_truth (pandas.DataFrame): The ground truth data.
    output_path (str): The path to save the output figure.

    Returns:
    None
    """
    df_ground_truth = df_ground_truth.rename(columns={'GFP_fluorescence': 'GFP fluorescence', 'RFP_fluorescence': 'mCherry fluorescence'})
    df_ground_truth = df_ground_truth.round(2)

    sns.set_theme(style="darkgrid")
    fig = sns.jointplot(x="GFP fluorescence", y="mCherry fluorescence", data=df_ground_truth, kind='reg', scatter_kws={'alpha': 0.1}, marginal_kws={'log_scale': True})
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=df_ground_truth['GFP fluorescence'], y=df_ground_truth['mCherry fluorescence'])

    print(p)
    plt.text(10, 1000, f'y = {round(intercept, 3)} + {round(slope, 3)}x\nR_squared = {round(r ** 2, 3)}')

    fig.ax_joint.set_ylim(ymin=1)
    fig.ax_joint.set_xscale('log')
    fig.ax_joint.set_yscale('log')
    fig.savefig(output_path)

def main():
    # Set working directory and load data
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, 'raw_data', 'cleaned_data.csv')
    df_ground_truth = load_data(data_path)

    # Plot data
    output_path = os.path.join(script_dir, 'figures', 'fluorescence_label_comparison.png')
    plot_data(df_ground_truth, output_path)

if __name__ == "__main__":
    main()