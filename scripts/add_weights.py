import datetime
import numpy as np
import uproot
import pandas as pd
import mplhep as hep
hep.style.use("LHCb2")
import argparse
import os
import awkward as ak
from pprint import pprint



def signalname_from_filename(file):
    if "Bu2JpsiK" in file:
        signalname = r"$B^+ \to J/\psi K^+$"
    if "Bd2JpsiKst" in file:
        signalname = r"$B^{*0} \to J/\psi K^*$"
    return signalname
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add the sample weights to the data set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_file', help='File with preselection applied')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--decayType', help='Event decay', type=str)
    parser.add_argument('--range', help='Observable range', nargs="+")
    parser.add_argument('--out_path', help='Where fit results and plots will be stored', type=str)
    parser.add_argument('--weight_file', help='Root file storing the weights', type=str)
    parser.add_argument('--loading_features', help='Path to file containing all features to load', type=str)



    cfg = parser.parse_args()
    pprint(cfg)
    mass_range = (int(cfg.range[0]), int(cfg.range[1]))

    #Load Dataset
    print(f'Loading dataset begins {datetime.datetime.now().strftime("%H:%M:%S")}', flush = True)
    
    with uproot.open(cfg.data_file) as _f:
        df_data = _f[cfg.treename].arrays(library="pd")    
    df_data.dropna(inplace=True)
    print(f'Loading dataset ends {datetime.datetime.now().strftime("%H:%M:%S")}', flush = True)
    print(df_data.columns, flush = True)


    df_data["event_entry"] =  df_data["file_id"].astype(str) + "_" + df_data["RUNNUMBER"].astype(str) + "_" + df_data["EVENTNUMBER"].astype(str)

    #Load Weightfile
    print(f'Loading weight file begins {datetime.datetime.now().strftime("%H:%M:%S")}', flush = True)
    with uproot.open(cfg.weight_file) as _f:
        df_weight = _f['DecayTree;1'].arrays(library="pd")
    print(f'Loading weight file ends {datetime.datetime.now().strftime("%H:%M:%S")}', flush = True)
    print(df_weight.columns, flush = True)
    df_weight.dropna(inplace=True)
    df_weight["event_entry"] =  df_weight["file_id"].astype(str) + "_" + df_weight["RUNNUMBER"].astype(str) + "_" + df_weight["EVENTNUMBER"].astype(str)

    df_data = df_data.reset_index()
    print(df_data.head(), flush = True)
    print(df_data.shape, flush = True)
    df_data.drop(columns=["level_0"], inplace=True)

    print(f"Before checking entries with weight {df_data.shape}", flush = True)
    df_data = df_data[df_data["event_entry"].isin(df_weight["event_entry"])] #Drops all events that have no sweight assigned by the mass_fit script
    print(f"After checking entries with weight {df_data.shape}", flush = True)


    if 'non_selected_signal_weights' in df_weight.columns:
        df_weight.rename(columns={'non_selected_signal_weights':     'signal_weights'},     inplace=True)
        df_weight.rename(columns={'non_selected_background_weights': 'background_weights'}, inplace=True)
    
    drop_cols = [col for col in df_weight.columns if col in df_data.columns]
    drop_cols.remove("event_entry")
    df_weight.drop(columns=drop_cols, inplace=True)

    print(f"Before merging {df_data.shape}", flush = True)
    df_data = df_data.merge(df_weight.reset_index(drop=True), on=["event_entry"], how='left')
    print(f"After merging {df_data.shape}", flush = True)
    print(df_data.columns, flush = True)
   
    print(df_data.head(), flush = True)
    print(df_data.shape, flush = True)

    df_data.drop(columns=['event_entry'], inplace=True)

    #Save weighted dataframe to disk
    tree_dict = {col: np.array(df_data[col]) for col in df_data.columns}

    for col in tree_dict:
        print(f'{col}: {tree_dict[col][0]} ({type(tree_dict[col][0])})', flush = True)

    path = os.path.join(cfg.out_path, os.path.basename(cfg.data_file))
    print(f'writing to {path}', flush = True)
    with uproot.recreate(path) as f:
        f['DecayTree'] = tree_dict
    
    print("Added weights", flush = True)