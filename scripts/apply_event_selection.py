import uproot
import pandas as pd
import numpy as np
import argparse
import os
from pprint import pprint
from scripts.train_BDT import KFoldBDT
import pickle
import yaml
import datetime

bin_var_translation = {'Tau': 'B_TAU'} #Expand here if other binning variables are added in the future

def load_bins(decay, binning, bin_file):
    with open(bin_file, 'r') as f:
        binnings = yaml.safe_load(f)

    for b in binnings[binning][decay]:
        if b == 'inf':
            b = np.inf
        else:
            b = float(b)
    
    return binnings[binning][decay]    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply signal selection to the provided Tuples as well as splitting file into bins for crosschecks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data', help='Path to the data file')
    parser.add_argument('--target', help='Path to the target file')
    parser.add_argument('--treename', help='Name of the tree in the ROOT file', default='DecayTree;1')
    parser.add_argument('--decay_type', help='Type of decay and the bin if applicable')
    parser.add_argument('--binning', help='Binning applied to the data if applicable, e.g. Tau1of5 for the first bin in a 5-bin split according to the Tau variable. Each bin contains roughly the same number of events.')
    parser.add_argument('--BDT', help='Path to the trained BDT model')
    parser.add_argument('--signal_class_features', help='Path to the yaml file containing the features used for training the BDT model')
    parser.add_argument('--bin_file', help='Path to the file containing the bin edges for the variable used for binning the data')

    cfg = parser.parse_args()
    pprint(cfg)
    os.makedirs(os.path.dirname(cfg.target), exist_ok=True)

    with open(cfg.BDT, 'rb') as f:
        loaded_data = pickle.load(f)
        BDT = loaded_data['model']
        cut   = loaded_data['cut']

    with uproot.open(cfg.data) as f:
        df = f[cfg.treename].arrays(library='pd')
    with open(cfg.signal_class_features, 'r') as f:
        signal_class_features = yaml.safe_load(f)[cfg.decay_type]
    
    df["event_entry"] = df["file_id"].astype(str) + "_" + df["RUNNUMBER"].astype(str) + "_" + df["EVENTNUMBER"].astype(str)
    print(f"Input file contains {df['event_entry'].nunique()} unique events", flush=True)

    # Apply the BDT to select signal events
    df_events = df[signal_class_features + ['event_entry']].groupby("event_entry").first().reset_index(drop=False)
    print(f"Prediction of signalness starts at {datetime.datetime.now()}", flush=True)
    df_events['signalness'] = BDT.predict_proba(df_events[signal_class_features].to_numpy(), df_events["event_entry"].values)
    print(f"Prediction of signalness ends at {datetime.datetime.now()}", flush=True)
    df_events = df_events[df_events['signalness'] > cut]
    df = df.merge(df_events[["event_entry", "signalness"]], on="event_entry", how="inner")
    del df_events

    print(f"Number of events after selection: {df['event_entry'].nunique()}", flush=True)

    #If the binning is provided, split the data into bins
    if cfg.binning is not None:
        print(f"Splitting the data into bins according to {cfg.binning}", flush=True)
        binning = cfg.binning.split("of")
        bin_idx = int(binning[0][-1])
        n_bins = int(binning[1])
        binning_var = bin_var_translation[binning[0][:-1]]
        bin_edges = load_bins(cfg.decay_type, binning[0][:-1], cfg.bin_file)

        for a,b in zip(bin_edges[1:], bin_edges[:-1]): #Check binning
            df_bin = df[df[binning_var].between(float(b), float(a))].groupby("event_entry").first().reset_index(drop=False)
            print(f"{len(df_bin)}", flush=True)

        df_events = df[['event_entry', binning_var]].groupby("event_entry").first().reset_index(drop=False)
        df_events = df_events[df_events[binning_var].between(float(bin_edges[bin_idx-1]), float(bin_edges[bin_idx]))]

        df = df.merge(df_events[["event_entry"]], on="event_entry", how="inner")
        del df_events
    
    with uproot.recreate(f"{cfg.target}") as file:
        file["DecayTree"] = df


