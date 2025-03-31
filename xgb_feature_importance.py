import pandas as pd
import os
from os import path
import seaborn as sns
import warnings

# Set plot sizes
sns.set_theme(rc={"figure.dpi": 600})
warnings.filterwarnings("ignore")

def main(shapley_path, output_path):
    """
    Main function to calculate and plot SHAP values for XGBoost models.

    Parameters:
    shapley_path (str): The path to the SHAP values.
    output_path (str): The path to save the output figures.

    Returns:
    None
    """
    os.makedirs(output_path, exist_ok=True)

    df_shapley = pd.read_csv(path.join(shapley_path, 'XGB_shapley_mean.csv'), index_col=0)
    df_shapley.loc[:, 'shapley_mean'] = df_shapley.mean(axis=1)
    df_shapley = df_shapley.sort_values('shapley_mean', ascending=False).head(20).round(2).reset_index(names='selected_feature')
    df_shapley_melt = df_shapley.melt(id_vars='selected_feature', value_vars=[str(x) for x in range(1, 21)], var_name='split', value_name='Shap_value')

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_shapley_melt, kind="bar", orient='h',
        y="selected_feature", x="Shap_value",
        errorbar="sd", palette="dark", alpha=.6, height=6, aspect=2
    )
    g.despine(left=True)
    g.set_axis_labels("Shapley Value", "Selected Features")
    g.set_titles("XGBoost Feature Importance")
    g.figure.savefig(path.join(output_path, 'XGB_shapley_bar_abs.png'))

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    shapley_path = os.path.join(script_dir, 'raw_data')
    output_path = os.path.join(script_dir, 'figures')
    main(shapley_path, output_path)