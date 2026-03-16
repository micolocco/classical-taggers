import json
import numpy as np
import os
import matplotlib.pyplot as plt
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
from scripts import utils
import pandas as pd
from matplotlib.patches import Patch
import shutil


def error_bar_plot(df, label, file):
    # Plot the results
    means = df.groupby(['Tagger', 'BN'])[label].mean().reset_index()
    stds = df.groupby(['Tagger', 'BN'])[label].std().reset_index()
    x = np.linspace(0, len(taggers)-1, len(taggers))
    y_noBN = np.array([means[(means['Tagger'] == tagger) & (means['BN'] == 'noBN')][label].iloc[0] for tagger in taggers])
    yerr = np.array([stds[(stds['Tagger'] == tagger) & (stds['BN'] == 'noBN')][label].iloc[0] for tagger in taggers])
    plt.errorbar(x-0.1, np.zeros_like(y_noBN), yerr=yerr, label='Baseline', color='blue', fmt='o', capsize=10)
    
    y = np.array([means[(means['Tagger'] == tagger) & (means['BN'] == 'noBN')][label].iloc[0] for tagger in taggers])
    yerr = np.array([stds[(stds['Tagger'] == tagger) & (stds['BN'] == 'noBN')][label].iloc[0] for tagger in taggers])    
    plt.errorbar(x+0.1, y-y_noBN, yerr=yerr, label='Batch Normalized', color='orange', fmt='o', capsize=10)

    plt.xticks(range(len(taggers)), taggers)
    plt.ylabel('difference in Tagging Power')
    plt.legend()
    plt.savefig(file)
    plt.clf()

def box_plot(df, label, file):
    means = df.groupby(['Tagger', 'BN'])[label].mean().reset_index()
    baseline_means = np.array([means[(means['Tagger'] == tagger) & (means['BN'] == 'noBN')][label].iloc[0] for tagger in taggers])


    box = []
    for tagger, mean in zip(taggers, baseline_means):
        box.append(df[(df['Tagger'] == tagger) & (df['BN'] == 'noBN')][label].values- mean)
        box.append(df[(df['Tagger'] == tagger) & (df['BN'] == 'BN')][label].values - mean)

    x_plot = []
    for i in range(len(taggers)):
        x_plot.append(i-0.1)
        x_plot.append(i+0.1)
    bplot = plt.boxplot(box, positions=x_plot, widths=0.15, showmeans=True, patch_artist=True)

    color = 'blue'
    for patch in bplot['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

        if color == 'blue':
            color = 'orange'
        else:
            color = 'blue'

    legend_handles = [
        Patch(facecolor='blue', alpha=0.5, label='Baseline'),
        Patch(facecolor='orange', alpha=0.5, label='Batch Normalized')
    ]

    plt.legend(handles=legend_handles)

    plt.xticks(range(len(taggers)), taggers)
    plt.ylabel('difference in Tagging Power')
    plt.grid()
    plt.savefig(file)
    plt.clf()

if __name__ == '__main__':
    taggers = ['OSKaon', 'OSMuon', 'OSElectron']
    configs = ['lr0.001_bs8192_nL6_nN64', 'lr0.001_bs8192_nL16_nN64']
    file_prefixes = ['6L64', '16L64']

    out_path = f"/ceph/users/togasa/FlavourTagging/MC/savedModels/control_plots/seeds/"

    for base_config, file_prefix in zip(configs, file_prefixes):
        taggingPower_df = pd.DataFrame(columns=['Tagger', 'Seed', 'BN', 'TaggingPower', 'TaggingPower_sig'])
        for tagger in taggers:
            base_path = f"/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/"
            final_path = f"/testing/Data/logit/taggingInfo_logit.json"
            print(f"Processing tagger {tagger} with base config {base_config}", flush=True)
            
            #get all seeds used, by looking at the folders in the base path
            seeds = [int(folder) for folder in os.listdir(base_path)]
            for seed in seeds:

                path = f"{base_path}{seed}/{base_config}{final_path}"   
                if os.path.exists(path):
                    data = utils.load_and_process_json(path)
                    tagging_power = data['TaggingPower_Cali']

                    taggingPower_df.loc[len(taggingPower_df)] = [tagger, seed, 'noBN', tagging_power.nominal_value, tagging_power.nominal_value/tagging_power.std_dev]

                else:
                    print(f"File not found for seed {seed} in tagger {tagger} at path {path}")

                path_bn = f"{base_path}{seed}/{base_config}_BN{final_path}"   
                if os.path.exists(path_bn):
                    data = utils.load_and_process_json(path_bn)
                    tagging_power = data['TaggingPower_Cali']
                    
                    taggingPower_df.loc[len(taggingPower_df)] = [tagger, seed, 'BN', tagging_power.nominal_value, tagging_power.nominal_value/tagging_power.std_dev]
                else:
                    print(f"File not found for seed {seed} in tagger {tagger} at path {path_bn}")

        error_bar_plot(taggingPower_df, 'TaggingPower', os.path.join(out_path, f'{file_prefix}_Batch_norm_comparison.pdf'))
        box_plot(taggingPower_df, 'TaggingPower', os.path.join(out_path, f'{file_prefix}_Batch_norm_comparison_box.pdf'))

        error_bar_plot(taggingPower_df, 'TaggingPower_sig', os.path.join(out_path, f'{file_prefix}_Sig_Batch_norm_comparison.pdf'))
        box_plot(taggingPower_df, 'TaggingPower_sig', os.path.join(out_path, f'{file_prefix}_Sig_Batch_norm_comparison_box.pdf'))
