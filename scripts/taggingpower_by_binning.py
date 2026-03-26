import numpy as np
import matplotlib.pyplot as plt
import argparse
from pprint import pprint
import os
import yaml
import pandas as pd
import scripts.utils as utils

label_by_binning = {
    'Tau': r'decay time [ps]',}

def round_to_significant_figures(x, sig_figs):
    if x == 0:
        return 0
    else:
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)

def get_ticks(bins):
    ticks = ['<' + str(round_to_significant_figures(bins[1], 2))]
    if len(bins)-1 > 2:
        for i in range(1, len(bins)-2):
            ticks.append(str(round_to_significant_figures(bins[i], 2)) + '-' + str(round_to_significant_figures(bins[i+1], 2)))
    ticks.append('>' + str(round_to_significant_figures(bins[-2], 2)))
    return ticks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the tagging power by binning for different taggers')
    parser.add_argument('--taggers',           help='List of taggers to compare',nargs='+', default = ['OSKaon', 'OSMuon', 'OSElectron'])
    parser.add_argument('--features',          help='Name of feauterset used for training the taggers', default = 'union_PROBNN')
    parser.add_argument('--seed',              help='seed used during training', default = 12, type=int)
    parser.add_argument('--config',            help='config used during training', default = 'lr0.0001_bs8192_nL8_nN32', type=str)
    parser.add_argument('--binning_file',      help='Path to the file containing the binning information', default = 'configs/binnings.yaml', type=str,)
    parser.add_argument('--binning_name',      help='Name of the binning to use', default = 'Tau', type=str,)
    parser.add_argument('--decay',             help='Decay to analyze', type=str,)
    parser.add_argument('--saved_models_path', help='Path to the saved models', type=str,)
    parser.add_argument('--cut',               help='Cut to apply on the data', type=str)
    cfg = parser.parse_args()
    pprint(cfg)

    outpath = os.path.dirname(cfg.saved_models_path)
    outpath = os.path.join(outpath, f'hyperparameters_plots/{cfg.cut}/{cfg.binning_name}_bins/{cfg.decay}')
    os.makedirs(outpath, exist_ok=True)

    bins = yaml.safe_load(open(cfg.binning_file, 'r'))[cfg.binning_name][cfg.decay]
    bins = np.array(bins, dtype=float)
    if bins[-1] == np.inf:
        bins[-1] = bins[-2]*2
    print(bins)

    # /union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json
    performances = pd.DataFrame(columns=['Tagger', 'Bin', 'Tagging_Power', 'Tagging_Power_unc'])
    for tagger in cfg.taggers:
        for i in range(len(bins)-1):
            bin_name = f'{cfg.decay}{cfg.binning_name}{i+1}of{len(bins)-1}'
            file_path = os.path.join(cfg.saved_models_path, f'{bin_name}/{tagger}/{cfg.cut}/{cfg.features}/{cfg.seed}/{cfg.config}/testing/Data/logit/taggingInfo_logit.json')

            data = utils.load_and_process_json(file_path)
            tagging_power, tagging_power_unc = data['TaggingPower_Cali'].nominal_value, data['TaggingPower_Cali'].std_dev

            performances.loc[len(performances)] = [tagger, bin_name, tagging_power, tagging_power_unc]

    for tagger in cfg.taggers:
        tagger_data = performances[performances['Tagger'] == tagger]
        # x = bins[:-1] + np.diff(bins)/2
        # xerr = np.diff(bins)/2
        xticks = get_ticks(bins)

        plt.errorbar(range(len(xticks)), tagger_data['Tagging_Power'], yerr=tagger_data['Tagging_Power_unc'], label=tagger, marker='o', capsize=5)
        plt.xticks(ticks = range(len(xticks)), labels = xticks)
        
        plt.xlabel(label_by_binning.get(cfg.binning_name, 'Binning'))
        plt.ylabel('Tagging Power')
        plt.title(f'{tagger}')
        plt.savefig(os.path.join(outpath, f'{tagger}_binned_tagging_power.pdf'), bbox_inches='tight')
        plt.clf()
    
    print(performances)
