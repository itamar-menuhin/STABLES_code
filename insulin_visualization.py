import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns
import scipy
import statsmodels.api as sm

# Set plot sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 12

def load_data(file_path):
    """
    Load the raw data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def process_data(raw_data):
    """
    Process the raw data by normalizing and filtering.

    Parameters:
    raw_data (pandas.DataFrame): The raw data.

    Returns:
    pandas.DataFrame: The processed data.
    """
    raw_data.loc[:, 'expression_levels'] = raw_data.expression_levels / 28.95
    raw_data = raw_data[raw_data.experiment != 'SUC2']
    raw_data.loc[:, 'Expression Levels [log scale]'] = np.log(raw_data.expression_levels)
    raw_data.loc[:, 'Time [Days]'] = raw_data["Time(Days)"]
    return raw_data

def plot_data(processed_data, output_path):
    """
    Plot the processed data and save the figure.

    Parameters:
    processed_data (pandas.DataFrame): The processed data.
    output_path (str): The path to save the output figure.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['P', 'X', 'o', 'v']
    decay_results = {}

    for ii, exp in enumerate(['ORIGINAL_SEQ', 'ESO', 'ARC15', 'CAF20']):
        df_curr = processed_data[processed_data.experiment == exp].copy()
        slope, intercept, r, p, sterr = scipy.stats.linregress(x=df_curr["Time [Days]"],
                                                               y=df_curr["Expression Levels [log scale]"])
        if exp == 'ORIGINAL_SEQ':
            exp = 'Original Sequence'
        if exp == 'ESO':
            exp = 'Sequence Optimization'

        sns.regplot(data=df_curr, x="Time [Days]", y="Expression Levels [log scale]", fit_reg=True, ax=ax,
                    label=f'{exp}, \ny={int(np.exp(intercept))}*exp({round(slope, 3)}T), R2={round(r ** 2, 2)}',
                    marker=markers[ii])
        X = sm.add_constant(df_curr['Time [Days]'])
        model = sm.OLS(df_curr['Expression Levels [log scale]'], X).fit()
        decay_results[exp] = {
            'slope': model.params[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared
        }
        print(f"\nExperiment {exp}:")
        print(f"Slope (decay rate): {decay_results[exp]['slope']:.4f}")
        print(f"P-value: {decay_results[exp]['p_value']:.8f}")
        print(f"R-squared: {decay_results[exp]['r_squared']:.4f}")

    ax.set_xlabel('Days in experiment')
    ax.set_ylabel('Log(Insulin titer in mg/L)')
    ax.legend()
    fig.savefig(output_path)

def main():
    # Set working directory and load data
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, 'raw_data', 'RAW_DATA_insulin.csv')
    raw_data = load_data(data_path)

    # Process data
    processed_data = process_data(raw_data)

    # Plot data
    output_path = os.path.join(script_dir, 'figures', 'insulin_results.png')
    plot_data(processed_data, output_path)

if __name__ == "__main__":
    main()