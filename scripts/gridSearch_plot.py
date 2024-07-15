import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import argparse
from IPython import embed

'''
python gridSearch_plot.py --tagger <tagger> --decayType <decay> --seed <seed> 
'''

# Function to propagate and round the errors and values
def propagate_and_round(values):
    values = np.array(values) * 100  # Multiply all values by 100

    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return [np.nan, np.nan]

    if len(values) > 2:  # For TaggingPower_Cali and EffectiveMistag_Cali
        combined_error = np.sqrt(np.sum(np.square(values[1:])))
        if np.isnan(combined_error) or np.isinf(combined_error):
            return [np.nan, np.nan]
        
        rounded_error = round(combined_error, -int(np.floor(np.log10(combined_error))))
        significant_digit = int(np.floor(np.log10(rounded_error)))
        rounded_value = round(values[0], -significant_digit)
        
        return [rounded_value, rounded_error]
    else:  # For other data
        max_error = max(values[1:])
        if np.isnan(max_error) or np.isinf(max_error):
            return [np.nan, np.nan]
        
        if max_error == 0:
            # If the maximum error is zero, no need to round it further
            significant_digit = 0
        else:
            significant_digit = int(np.floor(np.log10(max_error)))
        
        rounded_errors = [round(err, -significant_digit) if err != 0 else 0 for err in values[1:]]
        rounded_value = round(values[0], -significant_digit)
        
        return [rounded_value] + rounded_errors

# Function to format the annotations
def format_annotation(value_with_error):
    if np.isnan(value_with_error[0]):
        return "NaN"
    elif len(value_with_error) == 2:
        return f"{value_with_error[0]} ± {value_with_error[1]}"
    else:
        return f"{value_with_error[0]}"

# Define a function to plot heatmaps for each architecture
def plot_heatmaps(architecture, ax_before, ax_after, ax_logit):
    data_before = results_before[results_before['Architecture'] == architecture]
    data_after = results_mistag[results_mistag['Architecture'] == architecture]
    data_logit = results_logit[results_logit['Architecture'] == architecture]

    heatmap_data_before = data_before.pivot("Learning Rate", "Batch Size", "Tagging Power")
    heatmap_data_after = data_after.pivot("Learning Rate", "Batch Size", "Tagging Power")
    heatmap_data_logit = data_logit.pivot("Learning Rate", "Batch Size", "Tagging Power")

    annot_before = data_before.pivot("Learning Rate", "Batch Size", "Annotation")
    annot_after = data_after.pivot("Learning Rate", "Batch Size", "Annotation")
    annot_logit = data_logit.pivot("Learning Rate", "Batch Size", "Annotation")

    sns.heatmap(heatmap_data_before, ax=ax_before, cmap="YlGnBu", annot=annot_before, fmt="", cbar=False, vmin=vmin, vmax=vmax)
    sns.heatmap(heatmap_data_after, ax=ax_after, cmap="YlGnBu", annot=annot_after, fmt="", cbar=False, vmin=vmin, vmax=vmax)
    sns.heatmap(heatmap_data_logit, ax=ax_logit, cmap="YlGnBu", annot=annot_logit, fmt="", cbar=True, vmin=vmin, vmax=vmax)

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
    parser.add_argument('--decayType', help='Event decay', type=str, choices=('Bu2JpsiK, Bd2JpsiKst, Bs2DsPi, Bd2DPi'))
    parser.add_argument('--seed', help='Random seed', type=str,)

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
    results_folder = f"/ceph/users/molocco/Data/savedModels/withUT_MC_2024/{cfg.decayType}/{cfg.tagger}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/{cfg.seed}"

    for lr, bs, arch in combinations:
        folder_path = os.path.join(results_folder, f"lr{lr}_bs{bs}_{arch}")
        file_before = os.path.join(folder_path, "mistag/taggingInfo_mistag.json")
        file_mistag = os.path.join(folder_path, "mistag/taggingInfo_mistag.json")
        file_logit = os.path.join(folder_path, "logit/taggingInfo_logit.json")

        if os.path.exists(file_before):
            with open(file_before, 'r') as f:
                data_before = json.load(f)
                tagging_power_before = propagate_and_round(data_before['TaggingPower'])
                results_before.loc[(results_before['Learning Rate'] == lr) & 
                                (results_before['Batch Size'] == bs) & 
                                (results_before['Architecture'] == arch), 'Tagging Power'] = tagging_power_before[0]
                results_before.loc[(results_before['Learning Rate'] == lr) & 
                                (results_before['Batch Size'] == bs) & 
                                (results_before['Architecture'] == arch), 'Annotation'] = format_annotation(tagging_power_before)

        if os.path.exists(file_mistag):
            with open(file_mistag, 'r') as f:
                data_after = json.load(f)
                tagging_power_after = propagate_and_round(data_after['TaggingPower_Cali'])
                results_mistag.loc[(results_mistag['Learning Rate'] == lr) & 
                                (results_mistag['Batch Size'] == bs) & 
                                (results_mistag['Architecture'] == arch), 'Tagging Power'] = tagging_power_after[0]
                results_mistag.loc[(results_mistag['Learning Rate'] == lr) & 
                                (results_mistag['Batch Size'] == bs) & 
                                (results_mistag['Architecture'] == arch), 'Annotation'] = format_annotation(tagging_power_after)

        if os.path.exists(file_logit):
            with open(file_logit, 'r') as f:
                data_logit = json.load(f)
                tagging_power_logit = propagate_and_round(data_logit['TaggingPower_Cali'])
                results_logit.loc[(results_logit['Learning Rate'] == lr) & 
                                (results_logit['Batch Size'] == bs) & 
                                (results_logit['Architecture'] == arch), 'Tagging Power'] = tagging_power_logit[0]
                results_logit.loc[(results_logit['Learning Rate'] == lr) & 
                                (results_logit['Batch Size'] == bs) & 
                                (results_logit['Architecture'] == arch), 'Annotation'] = format_annotation(tagging_power_logit)
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
    output_path = f"/home/molocco/classical-taggers/tagging_power_tables/"
    if not os.path.exists(f"{output_path}/{cfg.seed}"):
        os.makedirs(f"{output_path}/{cfg.seed}")
    output_file = f"{output_path}/{cfg.seed}/{cfg.tagger}_GridSearch_table.pdf"    
    plt.savefig(output_file)
    print(f"Plot saved at {output_file}")
    #plt.show()
