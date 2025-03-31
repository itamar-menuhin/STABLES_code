import pandas as pd
import os
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

os.chdir('/mnt/c/Users/itamar/Desktop/pythonProject_model_improvement/data')

path_all_predictions = 'all_predictions_20seeds_lean.csv'

# df_all_preds = pd.read_csv(path_all_predictions, index_col =0).fillna('none')
#
# dict_rename_model = {'ElasticNet':['ElasticNet_KNN', 'ElasticNet_XGB'], 'KNN':['ElasticNet_KNN', 'KNN_XGB'], 'XGB':['ElasticNet_XGB', 'KNN_XGB']}
#
# list_df = []
#
# for mod in dict_rename_model.keys():
#     df_curr1 = df_all_preds[df_all_preds.model == mod]
#     df_curr2 = df_curr1.copy()
#
#     df_curr1.loc[:, 'model'] = dict_rename_model[mod][0]
#     list_df.append(df_curr1)
#     df_curr2.loc[:, 'model'] = dict_rename_model[mod][1]
#     list_df.append(df_curr2)

# df_all_preds = pd.concat(list_df)


def col_top_nperc(df, n, col):
    """

    :param df: data to be ranked
    :param n: top n% to be labeled as positive examples
    :param col: column to label by
    :return: data with top n% denoter column
    """

    num_genes = df.shape[0]
    df_ranked= df.sort_values(by = col, ascending = False).reset_index(drop = True).reset_index(names = 'index')
    df_ranked.loc[:, col+'_top'] = 0
    df_ranked.loc[df_ranked['index']<int((num_genes/100.0)*n), col+'_top'] = 1
    del df_ranked['index']
    return df_ranked

def col_top_n(df, n, col):
    """

    :param df: data to be ranked
    :param n: top n% to be labeled as positive examples
    :param col: column to label by
    :return: data with top n% denoter column
    """

    df_ranked= df.sort_values(by = col, ascending = False).reset_index(drop = True).reset_index(names = 'index')
    df_ranked.loc[:, col+'_top'] = 0
    df_ranked.loc[df_ranked['index']<n, col+'_top'] = 1
    del df_ranked['index']
    return df_ranked

def roc_auc_with_label(df_pred_label, n=5):

    """

    :param df_pred_label: data with prediction and label
    :param n: top n% are labeled as positives
    :return: ROC_AUC score for guessing top n% for prediction
    """

    df_pred_label = col_top_nperc(df_pred_label, n,  col='preds')
    df_pred_label = col_top_nperc(df_pred_label, n, col='label')

    score = metrics.roc_auc_score(df_pred_label['label_top'], df_pred_label['preds_top'])
    return score


def max_fluor_top3(df_pred_label):

    df_pred_label = df_pred_label.sort_values('preds', ascending = False)
    df_top = df_pred_label.head(3)
    score = df_top.label.max()
    return score

def quantile_fluor_top_i(df_pred_label, i =3):

    size_sample = df_pred_label.shape[0]
    df_preds_sort = df_pred_label.sort_values('preds', ascending = False)
    df_preds_sort = df_preds_sort.head(i)
    df_label_sort = df_pred_label.sort_values('label', ascending = False).reset_index(drop = True).reset_index(names = 'true_rank')
    all_cols = list(df_preds_sort)
    df_label_sort = df_label_sort.merge(df_preds_sort, on = all_cols)
    max_rank = df_label_sort.true_rank.min()
    score = 1 - (max_rank/size_sample)
    return score

def accuracy_top_recommendations(df_pred_label, num_top_recommendations = 3, percentile_to_consider_success =1):

    df_pred_label = col_top_n(df_pred_label, num_top_recommendations,  col='preds')
    df_pred_label = col_top_nperc(df_pred_label, percentile_to_consider_success, col='label')

    successes = float(df_pred_label[(df_pred_label.preds_top==1)&(df_pred_label.label_top==1)].shape[0])

    score = successes/num_top_recommendations
    return score

def corr_results(df_pred_label, method = 'spearman'):
    return df_pred_label.preds.corr(df_pred_label.label, method = method)

def bootstrap_analysis_single_model(df_pred_label, group_name, sample_size):
    """
    generate bootstrapped scores of predictions, for further analysis

    :param df_pred_label: data with prediction and label
    :param name: names of column in bootstrap analysis
    :return: bootstraped sanmples' scores for ndgc and ROC AUC
    """
    num_iters = df_pred_label.shape[0]
    list_bootstrap_samples = []
    if sample_size==0:
        sample_size=num_iters
    # size_sample = 500
    for ii in range(num_iters):
        if ii % 1000 == 0:
            print(ii)
        bootstrap_sample = df_pred_label.sample(sample_size, replace=True)
        max_top3 = max_fluor_top3(bootstrap_sample)
        quantile_top3 = quantile_fluor_top_i(bootstrap_sample, i = 3)
        quantile_top1 = quantile_fluor_top_i(bootstrap_sample, i = 1)

        auc_sample = roc_auc_with_label(bootstrap_sample)

        spearman_corr = corr_results(bootstrap_sample)
        pearson_corr = corr_results(bootstrap_sample, method='pearson')

        # accuracy31 = accuracy_top_recommendations(bootstrap_sample, num_top_recommendations = 3, percentile_to_consider_success =1)
        # accuracy51 = accuracy_top_recommendations(bootstrap_sample, num_top_recommendations = 5, percentile_to_consider_success =1)
        # accuracy33 = accuracy_top_recommendations(bootstrap_sample, num_top_recommendations = 3, percentile_to_consider_success =3)
        # accuracy53 = accuracy_top_recommendations(bootstrap_sample, num_top_recommendations = 5, percentile_to_consider_success =3)

        list_bootstrap_samples.append((auc_sample, max_top3, quantile_top3, quantile_top1, spearman_corr, pearson_corr))

    df_bootstrap = pd.DataFrame.from_records(list_bootstrap_samples, columns = ['auc', 'max_top3', 'quantile_top3', 'quantile_top1',  'spearman_corr', 'pearson_corr'])
    # df_bootstrap = df_bootstrap.mean().to_frame().T

    df_bootstrap.loc[:, 'model'] = group_name[0]
    df_bootstrap.loc[:, 'train_on'] = group_name[1]


    df_bootstrap = df_bootstrap[['model', 'train_on', 'auc', 'max_top3', 'quantile_top3', 'quantile_top1',  'spearman_corr', 'pearson_corr']]
    print(df_bootstrap)

    return df_bootstrap


def bootstrap_all_models(df, sample_size=0, mode = 'mean'):

    df_new = df.copy()
    if mode == 'mean':
        df_new = df_new.groupby(['model', 'train_on', 'gene'], as_index = False)[['preds', 'label']].mean()

    bootstrap_list = []
    groups_data = df_new.groupby(['model', 'train_on'])
    for group_name, df_curr in groups_data:
        bootstrap_curr = bootstrap_analysis_single_model(df_curr, group_name, sample_size)
        print(bootstrap_curr)

        bootstrap_list.append(bootstrap_curr)
    bootstrap_all = pd.concat(bootstrap_list, ignore_index = True)
    return bootstrap_all


def visualize_single_model(df, model = 'KNN_XGB', output_path = 'bootstrap_graphs'):

    df_new = df[df.model==model]
    df_new = df_new[df_new.train_on == 'all']

    print(df_new)
    fig, ax = plt.subplots()
    x= 'quantile_top3'
    sns.histplot(data=df_new, x=x, ax =ax, binwidth=0.005, stat = 'percent', binrange=(0.9, 1.0))
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, f'single_model_{x}.png'))

    fig, ax = plt.subplots()
    x= 'quantile_top1'
    sns.histplot(data=df_new, x=x, ax =ax, binwidth=0.005, stat = 'percent', binrange=(0.7, 1.0))
    plt.tight_layout()

    fig.savefig(os.path.join(output_path, f'single_model_{x}.png'))

    fig, ax = plt.subplots()
    x= 'spearman_corr'
    sns.histplot(data=df_new, x=x, ax =ax, binwidth=0.005, stat = 'percent', binrange=(0.46, 0.54))
    plt.tight_layout()

    fig.savefig(os.path.join(output_path, f'single_model_{x}.png'))




# df_bootstrap = bootstrap_all_models(df_all_preds, sample_size =6000)
# df_bootstrap.to_csv('bootstrap_20lean_6K_two_models.csv')
# df_bootstrap.to_csv('bootstrap_20lean_6K.csv')

# 0/0
list_df = []
df_bootstrap1 = pd.read_csv('bootstrap_20lean_6K.csv', index_col = 0)
list_df.append(df_bootstrap1)
# print(df_bootstrap.groupby(['train_on', 'model']).mean().round(2))

df_bootstrap2 = pd.read_csv('bootstrap_20lean_6K_two_models.csv', index_col = 0)
list_df.append(df_bootstrap2)
df_bootstrap = pd.concat(list_df, ignore_index=True)
# df_bootstrap = pd.read_csv('all_bootstrap_20seeds_1K.csv', index_col = 0)

# visualize_single_model(df_bootstrap2)

# print(df_bootstrap.groupby(['train_on', 'model']).mean().round(2))
#
df_temp = df_bootstrap2[df_bootstrap2.model == 'KNN_XGB']
df_temp = df_temp[df_temp.train_on == 'all']
#
print(df_temp.describe(percentiles = [0.025, 0.1, 0.15, 0.25, 0.3, 0.35, 0.4, 0.5, 0.975]))
#
# print(df_temp[df_temp.quantile_top3<0.5].shape[0]/df_temp.shape[0])
# print(df_temp[df_temp.quantile_top1<0.5].shape[0]/df_temp.shape[0])

0/0
# df_bootstrap1= df_bootstrap[df_bootstrap.model.isin(['XGB', 'KNN'])]
# print(df_bootstrap1.groupby(['train_on', 'model']).mean().round(2))
col = 'auc'
name_append = '_together'
# 0/0

# fig, ax = plt.subplots()
# sns.barplot(df_bootstrap, x="model", y=col, hue="train_on", ax = ax)
# ax.set_ylim([0.55, 0.75])
# fig.suptitle(col)
# fig.savefig(f'bootstrap_graphs/{col}{name_append}.png')

col = 'auc'
fig, ax = plt.subplots()
sns.boxplot(df_bootstrap[df_bootstrap.train_on =='all'], x="model", y=col, ax = ax)
fig.suptitle(col)
plt.xticks(rotation=15)

fig.savefig(f'bootstrap_graphs/{col}{name_append}.png')

# col = 'max_top3'
# fig, ax = plt.subplots()
# sns.barplot(df_bootstrap, x="model", y=col, hue="train_on", ax = ax)
# fig.suptitle(col)
# fig.savefig(f'bootstrap_graphs/{col}{name_append}.png')

col = 'max_top3'
fig, ax = plt.subplots()
sns.boxplot(df_bootstrap[df_bootstrap.train_on =='all'], x="model", y=col, ax = ax)
fig.suptitle(col)
plt.xticks(rotation=15)

fig.savefig(f'bootstrap_graphs/{col}{name_append}_all.png')

col = 'spearman_corr'
fig, ax = plt.subplots()
sns.boxplot(df_bootstrap[df_bootstrap.train_on =='all'], x="model", y=col, ax = ax)
fig.suptitle(col)
plt.xticks(rotation=15)

fig.savefig(f'bootstrap_graphs/{col}{name_append}_all.png')

col = 'pearson_corr'
fig, ax = plt.subplots()
sns.boxplot(df_bootstrap[df_bootstrap.train_on =='all'], x="model", y=col, ax = ax)
fig.suptitle(col)
plt.xticks(rotation=15)

fig.savefig(f'bootstrap_graphs/{col}{name_append}_all.png')


print(df_bootstrap[df_bootstrap.train_on =='all'])

col = 'quantile_top3'
fig, ax = plt.subplots()
sns.boxplot(df_bootstrap[df_bootstrap.train_on =='all'], x="model", y=col, ax = ax)
fig.suptitle(col)
plt.xticks(rotation=15)
plt.ylim([0.9, 1])

fig.savefig(f'bootstrap_graphs/{col}{name_append}_all.png')

col = 'quantile_top1'
fig, ax = plt.subplots()
sns.boxplot(df_bootstrap[df_bootstrap.train_on =='all'], x="model", y=col, ax = ax)
fig.suptitle(col)
plt.xticks(rotation=15)
plt.ylim([0.5, 1])

fig.savefig(f'bootstrap_graphs/{col}{name_append}_all.png')



def calculate_quantile_frequencies(df, model_col='model', quantile_col='quantile_top3'):
    """
    Calculate frequency of quantile ranges for each unique model.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    model_col : str, optional (default='model')
        Name of the column containing model categories
    quantile_col : str, optional (default='quantile_top3')
        Name of the column containing quantile values

    Returns:
    --------
    pandas.DataFrame
        A dataframe where:
        - Rows are unique models
        - Columns are quantile ranges
        - Values are frequencies of measurements in each quantile range
    """
    # Define quantile ranges
    quantile_ranges = [
        (0.0, 0.25, '0.0-0.25'),
        (0.25, 0.5, '0.25-0.5'),
        (0.5, 0.75, '0.5-0.75'),
        (0.75, 0.9, '0.75-0.9'),
        (0.9, 0.95, '0.9-0.95'),
        (0.95, 0.99, '0.95-0.99'),
        (0.99, 1.01, '0.99-1.0')
    ]

    # Prepare results dictionary
    results = {}

    # Group by model
    grouped = df.groupby(model_col)

    # Calculate frequencies for each model
    for model, group in grouped:
        model_frequencies = {}
        for low, high, range_name in quantile_ranges:
            # Calculate frequency for this specific quantile range
            freq = ((group[quantile_col] >= low) & (group[quantile_col] < high)).mean()
            model_frequencies[range_name] = freq

        results[model] = model_frequencies

    # Convert to DataFrame
    return pd.DataFrame.from_dict(results, orient='index')




def plot_quantile_distribution(freq_df, output_path, add_to_title = ', max on top 3 predictions',figsize=(12, 8), cmap='YlGnBu'):
    """
    Create a heatmap visualization of quantile frequencies across models.

    Parameters:
    -----------
    freq_df : pandas.DataFrame
        DataFrame with models as rows and quantile ranges as columns
    figsize : tuple, optional (default=(12, 8))
        Figure size for the plot
    cmap : str, optional (default='YlGnBu')
        Colormap for the heatmap

    Returns:
    --------
    matplotlib.figure.Figure
        The created visualization
    """
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(freq_df,
                     annot=True,  # Show numeric values in each cell
                     cmap=cmap,  # Color gradient
                     fmt='.1%',  # Format to 2 decimal places
                     cbar_kws={'label': 'Frequency'},
                     linewidths=0.5,  # Add lines between cells
                     square=True, ax=ax)  # Make cells square

    # Customize the plot
    plt.title('Quantile Distribution Across Models'+add_to_title, fontsize=16, pad=20)
    plt.xlabel('Quantile Ranges', fontsize=12)
    plt.ylabel('Models', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Tight layout to prevent cutting off labels
    plt.tight_layout()

    fig.savefig(output_path)
    return

df_compare = df_bootstrap[df_bootstrap.train_on =='all']

aa = calculate_quantile_frequencies(df_compare, quantile_col='quantile_top3')
plot_quantile_distribution(aa, f'bootstrap_graphs/top3_quantile_all.png', figsize=(12, 8), cmap='YlGnBu')

aa = calculate_quantile_frequencies(df_compare, quantile_col='quantile_top1')
plot_quantile_distribution(aa, f'bootstrap_graphs/top1_quantile_all.png', add_to_title=', top prediction', figsize=(12, 8), cmap='YlGnBu')

# col = 'accuracy33'
# fig, ax = plt.subplots()
# sns.boxplot(df_bootstrap[df_bootstrap.train_on =='all'], x="model", y=col, ax = ax)
# fig.suptitle(col)
# plt.xticks(rotation=15)
#
# fig.savefig(f'bootstrap_graphs/{col}{name_append}_all.png')
#
# col = 'accuracy31'
# fig, ax = plt.subplots()
# sns.boxplot(df_bootstrap[df_bootstrap.train_on =='all'], x="model", y=col, ax = ax)
# fig.suptitle(col)
# plt.xticks(rotation=15)
#
# fig.savefig(f'bootstrap_graphs/{col}{name_append}_all.png')
# col = 'max_top3'
# fig, ax = plt.subplots()
# sns.violinplot(df_bootstrap[df_bootstrap.train_on =='GFP'], x="model", y=col, hue="train_on", ax = ax)
# fig.suptitle(col)
# fig.savefig(f'bootstrap_graphs/{col}{name_append}_gfp.png')

# col = 'max_top3'
# fig, ax = plt.subplots()
# sns.violinplot(df_bootstrap[df_bootstrap.train_on =='RFP'], x="model", y=col, hue="train_on", ax = ax)
# fig.suptitle(col)
# fig.savefig(f'bootstrap_graphs/{col}{name_append}_rfp.png')
#
# curr_model = 'ElasticNet'
# df_bootstrap_curr = df_bootstrap[df_bootstrap['model'] == curr_model]
# for train_on in ['GFP', 'RFP', 'all']:
#     print(train_on)
#     print(df_bootstrap_curr[df_bootstrap_curr.train_on == train_on].describe())
#     print(df_bootstrap_curr[df_bootstrap_curr.train_on == train_on].sort_values(by = 'auc', ascending = False).head(10))
#     print(df_bootstrap_curr[df_bootstrap_curr.train_on == train_on].sort_values(by = 'max_top3', ascending = False).head(10))
#
# for col in ['features', 'pca', 'labelling']:
#     print(df_bootstrap_curr.groupby(['train_on', col])[['auc', 'max_top3']].mean())
# print(df_bootstrap)