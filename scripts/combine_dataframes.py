import numpy as np
import uproot
import pandas as pd
import os
import argparse
from pprint import pprint
import psutil


def read_files(files, treename, vars=None):
    df = None

    for i, f in enumerate(files):
        print(f"Reading input file {i+1}/{len(files)}: {f}", flush=True)
        print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB')
        

        with uproot.open("{}".format(f)) as _f:
            if vars is None:
                _df = _f[treename].arrays(library="pd")
            else:
                _df = _f[treename].arrays(vars, library="pd")

        _df.dropna(inplace = True)
        print(f"Number of tracks in file {i+1}: {_df.shape[0]}", flush=True)
        _df["event_entry"] = _df["file_id"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        _df['candidate_entry'] = _df['file_id'].astype(str) + "_" + _df['candidate_index'].astype(str)


        if 'selected' in _df.columns:
            #Drop rows which are not selected if selection has happened already. Keep one per unique event_entry to ensure correct efficiency calculation
            #Done here already to reduce memory usage of this step
            _df = pd.concat([
                _df[_df['selected'] == 1],
                _df[_df['selected'] == 0].drop_duplicates('event_entry')
            ]).reset_index(drop=True)

        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df], ignore_index = True)
        
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split each Data sample into training, validation and test sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_files', help='Files containing the data samples', nargs='+', type=str)
    parser.add_argument('--mc_files', help='Files containing the MC samples', nargs='+', type=str)
    parser.add_argument('--target_path', help='Name of the output dir', type=str)
    parser.add_argument('--treename', help='Tree name of the weighted ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--splits', help='Number of splits for the combined DataFrame', type=int, default=20)
    parser.add_argument('--evtType', help='Decay type of the samples, used for naming the output files', type=str)
    parser.add_argument('--loading_features', help='Path to file containing all features to load', type=str)
    parser.add_argument('--index', help='Starting Index of the subset of files to process, used for naming the output files', type=int)


    cfg = parser.parse_args()
    pprint(cfg)
    #Read all data files using uproot
    df_data = read_files(cfg.data_files, cfg.treename)

    if cfg.mc_files is not None: #Combine data and MC samples if MC files are provided. Used for training of domain adapted models

        #Read all MC files using uproot
        df_mc = read_files(cfg.mc_files, cfg.treename)
        df_mc.drop(columns=['B_Tr_T_Origin_Flag', 'B_BKGCAT'], inplace=True)

        df_data = df_data[:len(df_mc)] #Make sure not too much data in dataset

        #Rename columns to match
        rename_dict = {
            'B_ID' : 'B_TRUEID', #TODO change this B_ID and B_TRUEID should not be mixed for anything other then Bu
            'B_Tr_T_OWNPVIPSig' : 'B_Tr_T_BPVIPSig', 
            'B_Tr_T_absOWNPVIP' : 'B_Tr_T_absBPVIP', 
        }

        df_data.rename(columns=rename_dict, inplace=True)

        #Add domain column
        df_data['domain'] = 0  # Data domain
        df_mc['domain'] = 1    # MC domain

        # Combine data and MC
        print(f"Data shape: {df_data.shape}")
        print(f"MC shape: {df_mc.shape}")
        df_combined = pd.concat([df_data, df_mc], ignore_index=True)
        del df_data, df_mc  # Free memory
        print(f"Combined DataFrame shape: {df_combined.shape}")
    else: #If no MC files provided, just use the data. This is to combine many small data files into a manageable number of larger files, 
          #allowing working with subsets of the data during development and debugging without having to read the entire dataset .
        df_combined = df_data

    # Save combined DataFrame to disk
    if not os.path.exists(cfg.target_path):
        os.makedirs(cfg.target_path)


    # Split dataframe and save each split to a separate file
    # This allows training on subsets of the data during development and debugging without reading the entire dataset 
    # df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame
    
    
    # Shuffle dataframe without creating a copy of the entire dataframe in memory
    indices = np.random.RandomState(42).permutation(len(df_combined)) 
    df_combined = df_combined.iloc[indices].reset_index(drop=True)

    
    print(f"Total number of unique events in combined DataFrame: {df_combined['event_entry'].nunique()}", flush=True)
    print(f"Total number of unique candidates in combined DataFrame: {df_combined['candidate_entry'].nunique()}", flush=True)
    print(f"Total number of tracks in combined DataFrame: {len(df_combined)}", flush=True)
    

    event_entries = df_combined['event_entry'].unique()

    subsets = np.array_split(event_entries, cfg.splits)

    for i, subset in zip(range(cfg.index, cfg.index + len(subsets)), subsets):

        if cfg.splits > 1:
            subset_df = df_combined[df_combined['event_entry'].isin(subset)].reset_index(drop=True)
        else:
            subset_df = df_combined


        subset_df.drop(columns=['event_entry'], inplace=True)

        output_file = os.path.join(cfg.target_path, f'samples_{i}.root')
        print(f"Saving combined DataFrame subset to {output_file}", flush=True)
        tree_dict = {col: subset_df[col].to_numpy(copy=False) for col in subset_df.columns}

        with uproot.recreate(output_file) as f:
            f['DecayTree'] = tree_dict
        print(f'File written to {output_file}')
