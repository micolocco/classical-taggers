import json

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


'''
python scripts/taggingpower_by_binning.py 
--taggers OSKaon OSMuon OSElectron SSPion SSProton
--features union_PROBNN 
--trained_on Data
--link_func logit
--binning_file configs/binnings.yaml 
--binning_name Tau 
--decay Bd2JpsiKst 
--cut allBKGCAT_notSamePV_noOSP_SSK_balanced
--saved_models_path /ceph/users/togasa/FlavourTagging/Data/savedModels


python scripts/taggingpower_by_binning.py  --taggers OSKaon OSMuon OSElectron SSPion SSProton --features union_PROBNN  --trained_on Data --link_func logit --binning_file configs/binnings.yaml  --binning_name Tau  --decay Bd2JpsiKst  --cut allBKGCAT_notSamePV_noOSP_SSK_balanced --saved_models_path /ceph/users/togasa/FlavourTagging/Data/savedModels

python scripts/taggingpower_by_binning.py  --taggers OSKaon OSMuon OSElectron SSPion SSProton --features union_PROBNN  --trained_on Data --link_func logit --binning_file configs/binnings.yaml  --binning_name Tau  --decay Bd2JpsiKst  --cut notSamePV_noOSP --saved_models_path /ceph/users/togasa/FlavourTagging/Data/savedModels

python scripts/taggingpower_by_binning.py  --taggers OSKaon OSMuon OSElectron SSPion SSProton --features union_PROBNN_edited_for_benchmark  --trained_on Data --link_func logit --binning_file configs/binnings.yaml  --binning_name Tau  --decay Bd2JpsiKst  --cut allBKGCAT_notSamePV_noOSP_SSK_balanced --saved_models_path /ceph/users/togasa/FlavourTagging/MC/benchmarkModels --benchmark_version Run3v1

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(   description='Plot the tagging power by binning for different taggers')
    parser.add_argument('--taggers',           help='List of taggers to compare', nargs='+', default = ['OSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton'])
    parser.add_argument('--features',          help='Name of feauterset used for training the taggers', default = 'union_PROBNN')
    parser.add_argument('--cut',               help='Trackselection used to train the taggers', type=str)
    parser.add_argument('--trained_on',        help='Whether models were trained on MC or Data', type=str)
    parser.add_argument('--link_func',         help='Which link function to consider when extracting models to analyse', type=str, default = 'logit')
    parser.add_argument('--binning_file',      help='Path to the file containing the binning information', default = 'configs/binnings.yaml', type=str,)
    parser.add_argument('--binning_name',      help='Name of the binning to use', default = 'Tau', type=str,)
    parser.add_argument('--decay',             help='Decay to analyze', type=str, default = 'Bd2JpsiKst')
    parser.add_argument('--saved_models_path', help='Path to the saved models', type=str,)
    parser.add_argument('--benchmark_version', help='Optional argument, given when a benchmark version is to be tested. If yes add benchmark version name', type=str)
    cfg = parser.parse_args()
    pprint(cfg)

    outpath = os.path.dirname(cfg.saved_models_path)
    outpath = os.path.join(outpath, f'hyperparameters_plots/{cfg.binning_name}_bins/{cfg.features}/{cfg.cut}/{cfg.decay}')
    os.makedirs(outpath, exist_ok=True)

    bins = yaml.safe_load(open(cfg.binning_file, 'r'))[cfg.binning_name][cfg.decay]
    bins = np.array(bins, dtype=float)
    if bins[-1] == np.inf:
        bins[-1] = bins[-2]*2
    print(bins)

    with open(f'./best_tagger_candidates/{cfg.cut}/{cfg.trained_on}/candidatedTaggers_{cfg.link_func}.json', 'r') as f:
        best_taggers = json.load(f)

    performances = pd.DataFrame(columns=['Tagger', 'Bin', 'Tagging_Power', 'Tagging_Power_unc'])
    for tagger in cfg.taggers:
        for i in range(len(bins)-1):
            bin_name = f'{cfg.decay}{cfg.binning_name}{i+1}of{len(bins)-1}'

            if 'benchmarkModels' not in cfg.saved_models_path:
                config = f'lr{best_taggers[tagger]["learning_rate"]}_bs{best_taggers[tagger]["batch_size"]}_nL{best_taggers[tagger]["numlayers"]}_nN{best_taggers[tagger]["numneurons"]}'
                seed = best_taggers[tagger]['seed']

                file_path = os.path.join(cfg.saved_models_path, f'{bin_name}/{tagger}/{cfg.cut}/{cfg.features}/{seed}/{config}/testing/Data/{cfg.link_func}/taggingInfo_{cfg.link_func}.json')
            else:

                file_path = os.path.join(cfg.saved_models_path, f'{bin_name}/{tagger}/{cfg.benchmark_version}/testing/Data/{cfg.link_func}/taggingInfo_{cfg.link_func}.json')

            data = utils.load_and_process_json(file_path)
            tagging_power, tagging_power_unc = data['TaggingPower_Cali'].nominal_value, data['TaggingPower_Cali'].std_dev

            performances.loc[len(performances)] = [tagger, bin_name, tagging_power, tagging_power_unc]

        # Get unbinned performance
        file_path = file_path.replace(bin_name, cfg.decay)

        print(file_path)

        data = utils.load_and_process_json(file_path)
        tagging_power, tagging_power_unc = data['TaggingPower_Cali'].nominal_value, data['TaggingPower_Cali'].std_dev
        performances.loc[len(performances)] = [tagger, 'unbinned', tagging_power, tagging_power_unc]

    for tagger in cfg.taggers:
        tagger_data = performances[performances['Tagger'] == tagger]
        # x = bins[:-1] + np.diff(bins)/2
        # xerr = np.diff(bins)/2
        xticks = get_ticks(bins)

        unb_val = tagger_data[tagger_data['Bin'] == 'unbinned']['Tagging_Power'].values[0]
        unb_unc = tagger_data[tagger_data['Bin'] == 'unbinned']['Tagging_Power_unc'].values[0]
        tagger_data = tagger_data[tagger_data['Bin'] != 'unbinned']
        

        plt.errorbar(range(len(xticks)), tagger_data['Tagging_Power'], yerr=tagger_data['Tagging_Power_unc'], marker='o', capsize=5, linestyle='', label='Tagging power in each bin')
        plt.xticks(ticks = range(len(xticks)), labels = xticks)


        plt.fill_between([-1, len(xticks)+1], [unb_val - unb_unc, unb_val - unb_unc], [unb_val + unb_unc, unb_val + unb_unc], color='red', alpha=0.2, label='Unbinned tagging power ± unc')
        plt.axhline(unb_val, color='red', linestyle='--', label='Unbinned tagging power')


        
        plt.xlabel(label_by_binning.get(cfg.binning_name, 'Binning'))
        plt.xlim(-0.5, len(xticks)-0.5)
        plt.ylabel('Tagging Power [%]')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(outpath, f'{tagger}_binned_tagging_power.pdf'), bbox_inches='tight')
        print(f'Saved plot for {tagger} at {os.path.join(outpath, f"{tagger}_binned_tagging_power.pdf")}')
        plt.clf()
    
    print(performances)
