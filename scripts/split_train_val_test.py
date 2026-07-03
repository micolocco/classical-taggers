import numpy as np
import uproot
import pandas as pd
from IPython import embed
import os
import argparse
from pprint import pprint
import yaml
# Local import
import scripts.pyTorchTraining as pyTrain



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
    parser.add_argument('--to_split', help='File with preselection applied')
    parser.add_argument('--target_path', help='Name of the output dir', type=str)
    parser.add_argument('--treename', help='Name of the tree in the root file', type=str, default='DecayTree;1')
    parser.add_argument('--seed', help='Random seed', default=45, type = int) 
    parser.add_argument('--config', help='Config yaml', type=str, default='configs/hyperpar_intervals.yaml') 
    parser.add_argument('--data_type', help="Type of Data used, MC, Data or domain_adapted when using domain adaptation",choices=('MC', 'Data', 'domain_adapted'))
    parser.add_argument('--decay', help='Decay which is being used', type=str)


    cfg = parser.parse_args()
    pprint(cfg)

    with open(f'{cfg.config}', 'r') as file:
        config = yaml.safe_load(file)

    #Read File
    print(f"Reading file: {cfg.to_split}", flush=True)
    with uproot.open("{}".format(cfg.to_split)) as f:
        df = f[cfg.treename].arrays(library="pd")
    if cfg.data_type != 'domain_adapted': # for Domain adaptation nans are dropped previously and some columns are naturally assigned nans
        df.dropna(inplace = True)
    
    if cfg.data_type != 'domain_adapted':
        df["event_entry"] = df["file_id"].astype(str) + "_" + df["RUNNUMBER"].astype(str) + "_" + df["EVENTNUMBER"].astype(str)
    else:
        df.loc[df['domain'] == 0, 'event_entry'] = df["file_id"].astype(str) + "_" + "data" + "_" + df["RUNNUMBER"].astype(str) + "_" + df["EVENTNUMBER"].astype(str)
        df.loc[df['domain'] == 1, 'event_entry'] = df["file_id"].astype(str) + "_" + "mc"   + "_" + df["RUNNUMBER"].astype(str) + "_" + df["EVENTNUMBER"].astype(str)


    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)


    print(df.columns)
    print(df.head(10))

    

    print(f'Total: {len(df["event_entry"].unique())} events, {len(df)} tracks', flush=True)
    
    #Split into train, validation and test Dataframes
    split_dfs = list(pyTrain.splitByEvent(df=df, seed=cfg.seed, train_val_split=config['-train_val_split'], do_train_split = not (cfg.decay[:2] == 'Bs' and cfg.data_type == 'Data')))
    #Write each frame to file
    purposes = ['train', 'validation', 'test']
    # if : 
    #     # for Bs data, training is not needed as measured Bs data cannot be used for training, only for testing and maybe validation. For other decays, all splits are needed.
    #     # -> Absorb train split into testing in this case
    #     split_dfs[2] = pd.concat([split_dfs[0], split_dfs[2]], ignore_index=True) 

    #     # Create an empty train split to avoid issues with snakemake
    #     split_dfs[0] = pd.DataFrame(columns=split_dfs[0].columns)
        

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
            
            print('\n\n', flush=True)


        #Drop event_entry because uproot can't write arrays of strings to disk
        df_.drop(columns=['event_entry'], inplace = True)

        tree_dict = {col: df_[col].to_numpy(copy=False) for col in df_.columns}
        path = os.path.join(cfg.target_path, p)
        path = os.path.join(path, os.path.basename(cfg.to_split))
        with uproot.recreate(path) as f:
            f['DecayTree'] = tree_dict
        
        print(f'Saved {p} file to {path}')

