import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse
from pprint import pprint

def read_files(files, tree, vars, weights = None):
    df = None
    if weights is not None:
        with uproot.open(weights) as weights_file:
            weights_df = weights_file['DecayTree;1'].arrays(event_entry_vars+['signal_weights'], library='pd')
        weights_df["event_entry"] = weights_df["file_id"].astype(str) + "_" + weights_df["RUNNUMBER"].astype(str) + "_" + weights_df["EVENTNUMBER"].astype(str)
        weights_df.drop(columns=event_entry_vars, inplace=True)

    for file in files:
        print(f"Reading {file}...")
        with uproot.open(file) as f:
            loading_vars = vars + event_entry_vars

            _df = f[tree].arrays(loading_vars, library="pd")
            _df["event_entry"] = _df["file_id"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            _df.drop(columns=event_entry_vars, inplace=True)
            
            if weights is not None:
                _df = _df.merge(weights_df[['event_entry', 'signal_weights']], on='event_entry', how='inner')

            if df is None:
                df = _df
            else:
                df = pd.concat([df, _df], ignore_index=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ProbNN control plots in MC and Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--decay', type=str, default='Bu2JpsiK', help='Decay channel')
    parser.add_argument('--nbins', type=int, default=50, help='Number of bins for histograms')
    parser.add_argument('--outpath', type=str, default=f'/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/control_plots/', help='Output path for control plots')

    args = parser.parse_args()
    pprint(args)
    decay = args.decay
    nbins = args.nbins
    outpath = args.outpath

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    signal_class_features = '/home/togasa/classical-taggers/configs/signal_classifier_features.yaml'
    bdt_model = f'/ceph/users/togasa/FlavourTagging/Data/signal_classifier/{decay}/bdt_model.pkl'

    data_path = f'/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/2_split/{decay}/test/*.root'
    mc_path   = f'/ceph-kernel/users/togasa/FlavourTagging/MC/NTuples/2_split/{decay}/test/*.root'
    data_files = glob.glob(data_path)
    mc_files = glob.glob(mc_path)
    weight_file = f'/ceph-kernel/users/togasa/FlavourTagging/Data/mass_fit/{decay}/test/weights_selected.root'




    tree = 'DecayTree;1'
    vars = [
        'B_Tr_T_PROBNN_K',
        'B_Tr_T_PROBNN_P',
        'B_Tr_T_PROBNN_E',
        'B_Tr_T_PROBNN_MU',
        'B_Tr_T_PROBNN_PI',
        'B_Tr_T_PIDK',
        'B_Tr_T_PIDe',
        'B_Tr_T_PIDmu',
        'B_Tr_T_PIDP',
        'B_Tr_T_GHOSTPROB',
    ]
    event_entry_vars = ['file_id', 'RUNNUMBER', 'EVENTNUMBER']
        #Read all files and concatenate into a single DataFrame

    df_data = read_files(data_files, tree, vars, weight_file)
    df_mc = read_files(mc_files, tree, vars)

    plt.figure(figsize=(14, 6))
    for i, var in enumerate(vars):
        plt.subplot(2, 5, i+1)
        min_val = min(df_data[var].min(), df_mc[var].min())
        max_val = max(df_data[var].max(), df_mc[var].max())
        bins = np.linspace(min_val, max_val, nbins+1)


        plt.hist(df_data[var], bins=bins, alpha=0.5, label='Data', density=True, color = 'blue')
        plt.hist(df_mc[var], bins=bins, alpha=0.5, label='MC', density=True, color = 'orange')
        plt.yscale('log')
        plt.xlabel(var)
        plt.legend()
    outfile = os.path.join(outpath, f'{decay}_control_plots.pdf')
    print(f'Control plots saved at {outfile}')
    plt.savefig(outfile, bbox_inches='tight',)