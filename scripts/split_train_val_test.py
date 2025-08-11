import numpy as np
import uproot
import pandas as pd
from matplotlib import pyplot as plt
from IPython import embed
import os
import argparse
from pprint import pprint
import yaml
# Local import
import scripts.pyTorchTraining as pyTrain
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)


def ascii_histogram(data, bins=10, symbol='#'):
    counts, bin_edges = np.histogram(data, bins=bins)
    width = np.max(data)-np.min(data)/bins
    max_count = max(counts)
    
    for i in range(bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        count = counts[i]
        bar_len = int((count / max_count) * width)
        bar = symbol * bar_len
        print(f'{bin_start:>7.2f} - {bin_end:>7.2f} | {bar} ({count})', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split each Data sample into training, validation and test sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--weighted', help='File with preselection applied')
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='../test')
    parser.add_argument('--treename', help='Tree name of the weighted ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--seed', help='Random seed', default=45, type = int) 
    parser.add_argument('--decayType', help='Event decay', type=str)
    parser.add_argument('--config', help='Config yaml', type=str, default='configs/hyperpar_intervals.yaml') 
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton'))
    parser.add_argument('--data_type', help="Type of Data used, MC, Data or domain_adapted when using domain adaptation",choices=('MC', 'Data', 'domain_adapted'))

    cfg = parser.parse_args()
    pprint(cfg)

    with open(f'{cfg.config}', 'r') as file:
        config = yaml.safe_load(file)

    filename = os.path.basename(cfg.weighted)[:-5]
    if cfg.data_type != 'domain_adapted':
        if filename[-7:-2] == '.data':
            id = filename[:-7]
        else:
            id = filename[:-3]

    #Read File
    print(f"Reading file: {cfg.weighted}", flush=True)
    with uproot.open("{}".format(cfg.weighted)) as f:
        df = f[cfg.treename].arrays(library="pd")
    if cfg.data_type != 'domain_adapted': # for Domain adaptation nans are dropped previously and some columns are naturally assigned nans
        df.dropna(inplace = True)
    
    if cfg.data_type != 'domain_adapted':
        df["event_entry"] = id + "_" + df["RUNNUMBER"].astype(str) + "_" + df["EVENTNUMBER"].astype(str)
    else:
        # df["event_entry"] = df["file_id"].astype(str)

        df.loc[df['domain'] == 0, 'event_entry'] = df["file_id"].astype(str) + "_" + "data" + "_" + df["RUNNUMBER"].astype(str) + "_" + df["EVENTNUMBER"].astype(str)
        df.loc[df['domain'] == 1, 'event_entry'] = df["file_id"].astype(str) + "_" + "mc" + "_" + df["RUNNUMBER"].astype(str) + "_" + df["EVENTNUMBER"].astype(str)

    #Shuffle samples
    print(df.head(10))
    df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    print(df.head(10))

    # Assignation of the tagging decision (d)
    # d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
    if ("Bd" or "Bs" in cfg.decayType) and (cfg.tagger == "SSKaon" or cfg.tagger == "SSPion" ):
        df[f"{cfg.tagger}_TagDec"] = df[f"B_Tr_T_Charge"]
    else:
        df[f"{cfg.tagger}_TagDec"] = df[f"B_Tr_T_Charge"] * (-1)

    # Assignation of the label (it will be used as NN output)
    # The label is given by the product of the tagging decision and the flavour charge of the B.
    # It indicates if the tagging decision is wrong or correct.
    # -1 == wrong tag  1 == correct tag
    # When using data:
    #   - tagging decision: the B_TRUEID must be replaced with B_ID 
    #   - calibration: B_ID = reconstructed ID when moving to data!
    BID = 'B_ID' if cfg.data_type == 'Data' else 'B_TRUEID'

    df["label"] = df[f"{cfg.tagger}_TagDec"] * df[BID]/abs(df[BID]) 

    df.loc[df.label == -1, "label"] = 0 # shifting the label from -1 to 0

    #For domain adaptation data needs to unlabelled
    if cfg.data_type == 'domain_adapted':
        df.loc[df["domain"] == 0, "label"] = np.nan

    print(df.columns)
    print(df.head(10))


    print(f'Total: {len(df["event_entry"].unique())} events, {len(df)} tracks')

    #Split into train, validation and test Dataframes, train and validation only contains selected tracks
    split_dfs = pyTrain.splitByEvent(df=df, seed=cfg.seed, train_val_split=config['-train_val_split'])

    #Write each frame to file
    purposes = ['train', 'validation', 'test']
    for df_, p in zip(split_dfs, purposes):
        print(f'saving the {p} split. {len(df_["event_entry"].unique())} events, {len(df_)} tracks', flush=True)
        tracks_per_event = df_["event_entry"].value_counts().tolist()

        if len(tracks_per_event) > 0:
            print(f'minimum tracks per event: {np.min(tracks_per_event):.0f}')
            mean_tracks_per_event = np.mean(tracks_per_event)
            stdev_tracks_per_event = np.sqrt(np.mean((tracks_per_event - mean_tracks_per_event)**2))
            prop_dev_tracks_per_event =stdev_tracks_per_event/mean_tracks_per_event*100
            print(f'mean tracks per event: {mean_tracks_per_event:.2f} +/- {stdev_tracks_per_event:.2f} ({prop_dev_tracks_per_event:.2f}% relative standard deviation)')
            print(f'median tracks per event: {np.median(tracks_per_event):.0f}')
            
            print(f'maximum tracks per event: {np.max(tracks_per_event):.0f}')

            print('\n')

            ascii_histogram(tracks_per_event, bins = 14)
            
            print('\n\n')


        #Drop event_entry because uproot can't write arrays of strings to disk
        df_.drop(columns=['event_entry'], inplace = True)

        tree_dict = {col: np.array(df_[col]) for col in df_.columns}
        path = os.path.join(cfg.target_path, p)
        path = os.path.join(path, f'{filename}.root')
        with uproot.recreate(path) as f:
            f['DecayTree'] = tree_dict
        
        print(f'Saved {p} file to {path}')

