import json
import numpy as np
import argparse
from pprint import pprint
import os
import matplotlib.pyplot as plt
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import matplotlib.font_manager as fm
import plottingUtils
from IPython import embed

# Uncomment and adjust these lines if you need custom fonts
# font_path = '/home/lhcb/celani/miniconda3/envs/rkpipi/fonts/FreeSerifItalic.otf'
# fm.fontManager.addfont(font_path)
# plt.rcParams['text.usetex'] = False

tagging_perf_dict = {
    'TaggingEfficiency': r'$\varepsilon_{\mathrm{tag}}$',
    'TaggingEfficiency_Cali': r'$\varepsilon^{\mathrm{cali}}_{\mathrm{tag}}$',
    'EffectiveMistag': r'$\omega$',
    'EffectiveMistag_Cali': r'$\omega^{\mathrm{cali}}$',
    'TaggingPower': r'$\varepsilon_{tag,\mathrm{eff}}$',
    'TaggingPower_Cali': r'$\varepsilon_{tag,\mathrm{eff}}^{\mathrm{cali}}$'
}

def plot_tagging_power_vs_seed(seeds, calibrated_taggingPower, missing_seeds, lr, bs, arch):
    # Extract values and errors from the calibrated_taggingPower list
    values = [item[0] for item in calibrated_taggingPower]
    errors = [item[1] for item in calibrated_taggingPower]

    # Create the plot
    plt.errorbar(seeds, values, yerr=errors, fmt='o', capsize=5, label=f'Calibrated Tagging Power\nLR: {lr}, Arch: {arch}, BS: {bs}')    
    # Plot the missing data points with a red cross
    if missing_seeds:
        plt.scatter(missing_seeds, [0] * len(missing_seeds), color='red', marker='x', label='Not Found')

    plt.xlabel('Seed')
    plt.ylabel(f"{tagging_perf_dict['TaggingPower_Cali']} (%)")
    plt.title(f'{cfg.tagger}: Calibrated Tagging Power vs. Seed')
    plt.legend(fontsize=14, loc='lower right')   
    plt.grid(True)
    # Save the plot to a file
    output_path = f"/home/molocco/classical-taggers/seedPlots"
    if not os.path.exists(f"{output_path}"):
        os.makedirs(f"{output_path}")
    output_file = f"{output_path}/{cfg.tagger}.pdf"    
    plt.savefig(output_file)
    print(f"Plot saved at {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating all the output lines to be inserted in the Snakefile when all the configurations (in yaml format) in the configs folder are wanted',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--decayType', help='Event decay', type=str, choices=('Bu2JpsiK', 'Bd2JpsiKst', 'Bs2DsPi', 'Bd2DPi'))
    cfg = parser.parse_args()
    pprint(cfg)

    seeds = [2, 10, 12, 14, 45]
    learning_rates = [0.001,]
    batch_sizes = [32,]
    architectures = ['simple',]

    #learning_rates = [0.001, 0.01, 0.1]
    #batch_sizes = [32, 128, 1024, 2048]
    #architectures = ['simple', 'complex']

    for lr in learning_rates:
        for bs in batch_sizes:
            for arch in architectures:
                calibrated_taggingPower = []
                used_seeds = []
                missing_seeds = []
                for seed in seeds:
                    # Read tagging power values from JSON files
                    results_folder = f"/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024//{cfg.decayType}/{cfg.tagger}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/{seed}" #cfg.seed
                    folder_path = os.path.join(results_folder, f"lr{lr}_bs{bs}_{arch}")
                    json_file = os.path.join(folder_path, "mistag/taggingInfo_mistag.json")
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            tagging_power = plottingUtils.propagate_and_round(data['TaggingPower_Cali'])
                            if tagging_power:
                                calibrated_taggingPower.append(tagging_power)
                                used_seeds.append(seed)
                    else:
                        missing_seeds.append(seed)
                
                if calibrated_taggingPower or missing_seeds:
                    plot_tagging_power_vs_seed(used_seeds, calibrated_taggingPower, missing_seeds, lr, bs, arch)
