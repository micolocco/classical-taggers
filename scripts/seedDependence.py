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


def box_plot(df, label, file):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    x_plot = list(range(4))

    for ax, tagger in zip(axs, ['OSKaon', 'OSMuon', 'OSElectron']):
        ax.set_xticks([1.5, 3.5])
        ax.set_xticklabels(['Baseline', 'Batch Normalized'])
        ax.set_title(f'Tagger')
        ax.set_ylabel(label)
        ax.grid()

        df_tagger = df[df['Tagger'] == tagger]
        boxes = []
        boxes.append(df_tagger[(df_tagger['architecture'] == '6L64N') & (df_tagger['BN'] == 'noBN')][label].values)
        boxes.append(df_tagger[(df_tagger['architecture'] == '6L64N') & (df_tagger['BN'] == 'BN')][label].values)
        boxes.append(df_tagger[(df_tagger['architecture'] == '16L64N') & (df_tagger['BN'] == 'noBN')][label].values)
        boxes.append(df_tagger[(df_tagger['architecture'] == '16L64N') & (df_tagger['BN'] == 'BN')][label].values)

            
        bplot = ax.boxplot(boxes, positions=x_plot, widths=0.15, showmeans=True, patch_artist=True)

        color = 'blue'
        alpha = [0.4, 0.4, 0.7, 0.7]
        for patch, a in zip(bplot['boxes'], alpha):
            patch.set_facecolor(color)
            patch.set_alpha(a)

            if color == 'blue':
                color = 'orange'
            else:
                color = 'blue'

    legend_handles = [
        Patch(facecolor='blue', alpha=0.4, label='Baseline 6L64N'),
        Patch(facecolor='orange', alpha=0.4, label='Batch Normalized 6L64N'),
        Patch(facecolor='blue', alpha=0.7, label='Baseline 16L64N'),
        Patch(facecolor='orange', alpha=0.7, label='Batch Normalized 16L64N')
    ]

    axs[3].axis('off')
    axs[3].legend(handles=legend_handles, loc='center')

    plt.savefig(file)
    plt.clf()

if __name__ == '__main__':
    taggers = ['OSKaon', 'OSMuon', 'OSElectron']
    configs = ['lr0.001_bs8192_nL6_nN64', 'lr0.001_bs8192_nL16_nN64']
    file_prefixes = ['6L64N', '16L64N']


    out_path = f"/ceph-kernel/users/togasa/FlavourTagging/MC/savedModels/control_plots/seeds/"
    taggingPower_df = pd.DataFrame(columns=['Tagger', 'architecture', 'Seed', 'BN', 'TaggingPower', 'TaggingPower_sig'])
    for base_config, file_prefix in zip(configs, file_prefixes):
        for tagger in taggers:
            base_path = f"/ceph-kernel/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/"
            final_path = f"/testing/Data/logit/taggingInfo_logit.json"
            print(f"Processing tagger {tagger} with base config {base_config}", flush=True)
            
            #get all seeds used, by looking at the folders in the base path
            seeds = [int(folder) for folder in os.listdir(base_path)]
            if 15 in seeds:
                seeds.remove(15) 
            for seed in seeds:

                path = f"{base_path}{seed}/{base_config}{final_path}"   
                if os.path.exists(path):
                    data = utils.load_and_process_json(path)
                    tagging_power = data['TaggingPower_Cali']

                    taggingPower_df.loc[len(taggingPower_df)] = [tagger, file_prefix, seed, 'noBN', tagging_power.nominal_value, tagging_power.nominal_value/tagging_power.std_dev]

                else:
                    print(f"File not found for seed {seed} in tagger {tagger} at path {path}")

                path_bn = f"{base_path}{seed}/{base_config}_BN{final_path}"   
                if os.path.exists(path_bn):
                    data = utils.load_and_process_json(path_bn)
                    tagging_power = data['TaggingPower_Cali']
                    
                    taggingPower_df.loc[len(taggingPower_df)] = [tagger, file_prefix, seed, 'BN', tagging_power.nominal_value, tagging_power.nominal_value/tagging_power.std_dev]
                else:
                    print(f"File not found for seed {seed} in tagger {tagger} at path {path_bn}")

    box_plot(taggingPower_df, 'TaggingPower', os.path.join(out_path, f'{file_prefix}_Batch_norm_comparison_box.pdf'))
    box_plot(taggingPower_df, 'TaggingPower_sig', os.path.join(out_path, f'{file_prefix}_Sig_Batch_norm_comparison_box.pdf'))
