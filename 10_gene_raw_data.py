import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os 
import seaborn as sns
import warnings

# Set plot sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

sns.set_theme(rc={"figure.dpi": 600})
warnings.filterwarnings("ignore")
 
def calculate_pvalue_between_experiments(df, exp1, exp2):
    """
    Calculate the p-value for the difference in mean ratios between two experiments using a t-test.

    Parameters:
    df (pandas DataFrame): A DataFrame containing the experiment data with columns 'experiment', 'mean_ratio', and 'std_error'.
    exp1 (str): The name of the first experiment.
    exp2 (str): The name of the second experiment.

    Returns:
    float: The p-value for the difference in mean ratios between exp1 and exp2.
    """
    ratio_exp1 = df[df.experiment == exp1].iloc[0, 1]
    error_exp1 = df[df.experiment == exp1].iloc[0, 2]
    ratio_exp2 = df[df.experiment == exp2].iloc[0, 1]
    error_exp2 = df[df.experiment == exp2].iloc[0, 2]
    tot_err = ((error_exp1 ** 2) + (error_exp2 ** 2)) ** 0.5
    t_stat = np.abs(ratio_exp1 - ratio_exp2) / tot_err
# One-sample t-test to compare the difference in mean ratios
    p_value = 1 - stats.t.cdf(t_stat, 1)
    return p_value

def compute_mean_ratios_and_ttest(dataframe, output_path):
        """
    Calculate the mean ratio and standard error of the ratios between final and initial normalized fluorescence values
for each experiment. Perform a t-test to compare the mean ratio of GFP with the mean ratio of other experiments.

    Parameters:
    dataframe (pandas DataFrame): A DataFrame containing the experiment data with columns 'experiment', 'time', 'sample',
and 'normalized_fluorescence'.
    output_path (str): The path where the generated graph and CSV file will be saved.

    Returns:
    None
    """
    experiments = dataframe['experiment'].drop_duplicates()
    results = []

    for experiment in experiments:
        # Subset data for the current experiment
        curr_data = dataframe[dataframe['experiment'] == experiment]
        curr_data = curr_data[['time', 'sample', 'normalized_fluorescence']].dropna()

        # Separate initial and final values
        initial_values = curr_data[curr_data['time'] == 1]['normalized_fluorescence'].values
        final_values = curr_data[curr_data['time'] == 15]['normalized_fluorescence'].values

        # Compute ratios
        ratios = final_values / initial_values

        # Calculate mean and standard error of the ratios
        mean_ratio = np.mean(ratios)
        std_error = np.std(ratios) / np.sqrt(len(ratios))

        # Store results
        results.append({'experiment': experiment, 'mean_ratio': mean_ratio, 'std_error': std_error})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).sort_values('mean_ratio', ascending=True)

# Perform a t-test to compare GFP with other experiments
    ratio_GFP = results_df[results_df.experiment == 'GFP'].iloc[0, 1]
    error_GFP = results_df[results_df.experiment == 'GFP'].iloc[0, 2]
    ratio_coupled = results_df[results_df.experiment != 'GFP'].mean_ratio.mean()
    var_coupled = (results_df[results_df.experiment != 'GFP'].std_error ** 2).sum() / 100
    err_coupled = var_coupled ** 0.5
    tot_err = ((err_coupled ** 2) + (error_GFP ** 2)) ** 0.5
    t_stat = (ratio_coupled - ratio_GFP) / tot_err
# One-sample t-test for the difference in mean ratios
    p_value = stats.t.cdf(t_stat, 9)

    print(f"T-statistic: {t_stat}")
    print(f"P-value (1-tailed): {1 - p_value}")

# Generate p-values for each gene compared to CDC9
    list_genes = list(results_df.experiment)
    for gene in list_genes:
        print(f"P-value for CDC9 vs {gene}: {calculate_pvalue_between_experiments(results_df, 'CDC9', gene)}")

    # Plot bar graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = plt.bar(results_df['experiment'], results_df['mean_ratio'], yerr=results_df['std_error'],
                 capsize=5, color='skyblue')
    plt.xlabel('Experiment')
    plt.ylabel('Mean Ratio (Final / Initial)')
    plt.title('Mean Ratio for 10-Gene Experiment')
    plt.xticks(rotation=45)
    plt.ylim([0.0, 1.2])
    plt.tight_layout()
    fig.savefig(output_path + '.png')

    results_df.sort_values('mean_ratio').to_csv(output_path + '.csv')

    return
   
def perform_kruskal_wallis_test(data, time_threshold):
    """
    Analyze the normalized fluorescence data at a specified time threshold using the Kruskal-Wallis H-test.

    Parameters:
    data (pandas DataFrame): A DataFrame containing the experiment data with columns 'time', 'sample', 'experiment',
        and 'normalized_fluorescence'.
    time_threshold (int or float): The time threshold for filtering the data. Only data points with time >= time_threshold
        will be considered.

    Returns:
    None
    """
    # Filter for later time points
    late_data = data[data['time'] >= time_threshold].copy()

    late_data['intervention'] = late_data.experiment
    late_data['value'] = late_data.normalized_fluorescence

    # Compare between all interventions using the Kruskal-Wallis H-test
    intervention_groups = [group['value'].values
                           for name, group in late_data
                           .groupby('intervention')]

    _, p_value_between_interventions = stats.kruskal(*intervention_groups)
    print(f"P-value for Kruskal-Wallis H-test: {p_value_between_interventions}")

    return

def normalize_fluorescence_data(df):
    """
    Normalize the fluorescence data by subtracting the mean fluorescence of the WT (wild-type) samples
    at each time point and dividing by the fluorescence value of the corresponding WT sample at time point 1.

    Parameters:
    df (pandas DataFrame): A DataFrame containing the experiment data with columns 'time', 'sample', 'experiment',
        and 'fluorescence'. The DataFrame should include data for both WT and experimental samples.

    Returns:
    pandas DataFrame: A new DataFrame with an additional column 'normalized_fluorescence'.
    """
    df_WT = df[df.experiment == 'WT']
    df_WT = df_WT[df_WT['sample'] != 3]
    df_WT = df_WT.groupby('time', as_index=False).fluorescence.mean().rename(columns={'fluorescence': 'WT'})

    df_exps = df[df.experiment != 'WT']
    df_exps = df_exps.merge(df_WT, on='time')
    df_exps.loc[:, 'fluorescence'] = df_exps.fluorescence - df_exps.WT

    normalization_values = df_exps[df_exps['time'] == 1].set_index(['experiment', 'sample'])['fluorescence']

    def normalize(group):
        group['normalized_fluorescence'] = group.apply(
            lambda row: row['fluorescence'] / normalization_values.loc[(row['experiment'], row['sample'])],
            axis=1
        )
        return group

    normalized_df = df_exps.groupby(['experiment', 'sample'], group_keys=False).apply(normalize)
    return normalized_df


# Set working directory and load data
script_dir = os.path.dirname(__file__)
normalized_fluorescence_path = os.path.join(script_dir, 'raw_data', 'normalized_fluorescence.csv')
df_fluor = pd.read_csv(normalized_fluorescence_path)

# Normalize fluorescence data
df_fluor = pd.melt(df_fluor, id_vars=['time', 'sample'], var_name='experiment', value_name='fluorescence')
df_fluor = normalize_fluorescence_data(df_fluor)

# Fit ratio and analyze endpoints
compute_mean_ratios_and_ttest(df_fluor, os.path.join(script_dir, 'processed_data', 'ratios'))
perform_kruskal_wallis_test(df_fluor, 15)
