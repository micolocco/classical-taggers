import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import os
import argparse
from pprint import pprint
'''
python gridSearch_Plot.py --tagger <tagger> --decayType <decay>
'''

# Define a function to plot heatmaps for each architecture
def plot_heatmaps(architecture, ax_before, ax_after, ax_logit):
    data_before = results_before[results_before['Architecture'] == architecture]
    data_after = results_mistag[results_mistag['Architecture'] == architecture]
    data_logit = results_logit[results_logit['Architecture'] == architecture]

    heatmap_data_before = data_before.pivot("Learning Rate", "Batch Size", "Tagging Power")
    heatmap_data_after = data_after.pivot("Learning Rate", "Batch Size", "Tagging Power")
    heatmap_data_logit = data_logit.pivot("Learning Rate", "Batch Size", "Tagging Power")

    sns.heatmap(heatmap_data_before, ax=ax_before, cmap="YlGnBu", annot=True, fmt=".3f", cbar=False, vmin=vmin, vmax=vmax)
    sns.heatmap(heatmap_data_after, ax=ax_after, cmap="YlGnBu", annot=True, fmt=".3f", cbar=False, vmin=vmin, vmax=vmax)
    sns.heatmap(heatmap_data_logit, ax=ax_logit, cmap="YlGnBu", annot=True, fmt=".3f", cbar=True, vmin=vmin, vmax=vmax)

    ax_before.set_title(f'Tagging Power Before Calibration ({architecture})', fontsize=10)
    ax_after.set_title(f'Calibrated Tagging Power with mistag ({architecture})', fontsize=10)
    ax_logit.set_title(f'Calibrated Tagging Power with Logit({architecture})', fontsize=10)
    ax_before.set_ylabel('Learning Rate')
    ax_after.set_ylabel('Learning Rate')
    ax_logit.set_ylabel('Learning Rate')
    ax_before.set_xlabel('Batch Size')
    ax_after.set_xlabel('Batch Size')
    ax_logit.set_xlabel('Batch Size')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating all the output lines to be inserted in the Snakefile when all the configurations (in yaml format) in the configs folder are wanted',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--decayType', help='Event decay', type=str)
    cfg = parser.parse_args()
    # Define the hyperparameter values
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [32, 128, 1024]
    architectures = ['simple', 'complex']

    # Generate all possible combinations of hyperparameters
    combinations = list(product(learning_rates, batch_sizes, architectures))

    # Create DataFrames to hold the results
    results_before = pd.DataFrame(combinations, columns=['Learning Rate', 'Batch Size', 'Architecture'])
    results_mistag = pd.DataFrame(combinations, columns=['Learning Rate', 'Batch Size', 'Architecture'])
    results_logit = pd.DataFrame(combinations, columns=['Learning Rate', 'Batch Size', 'Architecture'])

    # Read tagging power values from JSON files
    results_folder = f"/ceph/users/molocco/Data/savedModels/withUT_MC_2024/{cfg.decayType}/{cfg.tagger}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2"

    for lr, bs, arch in combinations:
        folder_path = os.path.join(results_folder, f"lr{lr}_bs{bs}_{arch}")
        file_before = os.path.join(folder_path, "mistag/taggingInfo_mistag.json")
        file_mistag = os.path.join(folder_path, "mistag/taggingInfo_mistag.json")
        file_logit = os.path.join(folder_path, "logit/taggingInfo_logit.json")

        if os.path.exists(file_before):
            with open(file_before, 'r') as f:
                data_before = json.load(f)
                tagging_power_before = data_before['TaggingPower']
                results_before.loc[(results_before['Learning Rate'] == lr) & 
                                (results_before['Batch Size'] == bs) & 
                                (results_before['Architecture'] == arch), 'Tagging Power'] = tagging_power_before[0]

        if os.path.exists(file_mistag):
            with open(file_mistag, 'r') as f:
                data_after = json.load(f)
                tagging_power_after = data_after['TaggingPower_Cali']
                results_mistag.loc[(results_mistag['Learning Rate'] == lr) & 
                                (results_mistag['Batch Size'] == bs) & 
                                (results_mistag['Architecture'] == arch), 'Tagging Power'] = tagging_power_after[0]

        if os.path.exists(file_logit):
            with open(file_logit, 'r') as f:
                data_logit = json.load(f)
                tagging_power_logit = data_logit['TaggingPower_Cali']
                results_logit.loc[(results_logit['Learning Rate'] == lr) & 
                                (results_logit['Batch Size'] == bs) & 
                                (results_logit['Architecture'] == arch), 'Tagging Power'] = tagging_power_logit[0]
        else:
            results_logit.loc[(results_logit['Learning Rate'] == lr) & 
                            (results_logit['Batch Size'] == bs) & 
                            (results_logit['Architecture'] == arch), 'Tagging Power'] = np.nan

    # Convert tagging power to percentage and round to 3 decimal places
    results_before['Tagging Power'] *= 100
    results_before['Tagging Power'] = results_before['Tagging Power'].round(3)
    results_mistag['Tagging Power'] *= 100
    results_mistag['Tagging Power'] = results_mistag['Tagging Power'].round(3)
    results_logit['Tagging Power'] *= 100
    results_logit['Tagging Power'] = results_logit['Tagging Power'].round(3)

    # Determine the range for the color bar scale
    vmin = min(results_before['Tagging Power'].min(), results_mistag['Tagging Power'].min(), results_logit['Tagging Power'].min())
    vmax = max(results_before['Tagging Power'].max(), results_mistag['Tagging Power'].max(), results_logit['Tagging Power'].max())

    # Create a plot with multiple heatmaps
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

    # Plot heatmaps for each architecture
    plot_heatmaps('simple', axes[0, 0], axes[0, 1], axes[0, 2])
    plot_heatmaps('complex', axes[1, 0], axes[1, 1], axes[1, 2])

    # Add the main title
    fig.suptitle(f'Tagging Power Comparison for {cfg.tagger} in {cfg.decayType}', fontsize=16)

    # Adjust layout to make room for the main title
    plt.tight_layout(rect=[0, 0, .9, 0.95])
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3) 
    # Save the plot to a file
    output_file = f"/ceph/users/molocco/Data/savedModels/withUT_MC_2024/{cfg.decayType}/{cfg.tagger}_GridSearch_table.pdf"
    plt.savefig(output_file)
    print(f"Plot saved at {output_file}")
    #plt.show()
