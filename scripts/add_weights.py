import numpy as np
import uproot
import pandas as pd
import mplhep as hep
hep.style.use("LHCb2")
import argparse
import os
import awkward as ak
from pprint import pprint


from scripts.adding_features import get_loading_vars

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
    parser.add_argument('--obs_name', help='Name of observable', type=str, default="B_DTF_PV_Jpsi_MASS")
    parser.add_argument('--range', help='Observable range', nargs="+")
    parser.add_argument('--out_path', help='Where fit results and plots will be stored', type=str)
    parser.add_argument('--weight_file', help='Root file storing the weights', type=str)
    parser.add_argument('--loading_features', help='Path to file containing all features to load', type=str)



    cfg = parser.parse_args()
    pprint(cfg)
    mass_range = (int(cfg.range[0]), int(cfg.range[1]))

    #Load Dataset
    with uproot.open(cfg.data_file) as _f:
        df_data = _f[cfg.treename].arrays(library="pd")    
    df_data.dropna(inplace=True)


    df_data["event_entry"] =  df_data["file_id"].astype(str) + "_" + df_data["RUNNUMBER"].astype(str) + "_" + df_data["EVENTNUMBER"].astype(str)

    #Load Weightfile
    with uproot.open(cfg.weight_file) as _f:
        df_weight = _f['DecayTree;1'].arrays(library="pd")
    df_weight.dropna(inplace=True)
    df_weight["event_entry"] =  df_weight["file_id"].astype(str) + "_" + df_weight["RUNNUMBER"].astype(str) + "_" + df_weight["EVENTNUMBER"].astype(str)

    df_data = df_data.reset_index()
    print(df_data.head())
    print(df_data.shape)
    df_data.drop(columns=["level_0"], inplace=True)
    df_data = df_data[df_data["event_entry"].isin(df_weight["event_entry"])] #Drops all events that have not been selected in the mass_fit script



    df_weight.drop(columns=["B_DTF_PV_Jpsi_MASS", 'RUNNUMBER', 'EVENTNUMBER', "B_ID", "file_id", "index"], inplace=True)
    df_data = df_data.merge(df_weight.reset_index(drop=True), on=["event_entry"], how='left')
    print(df_data.columns)
   
    print(df_data.head())
    print(df_data.shape)

    df_data.drop(columns=['event_entry'], inplace=True)

    #Save weighted dataframe to disk
    tree_dict = {col: np.array(df_data[col]) for col in df_data.columns}

    for col in tree_dict:
        print(f'{col}: {tree_dict[col][0]} ({type(tree_dict[col][0])})')

    path = os.path.join(cfg.out_path, os.path.basename(cfg.data_file))
    print(f'writing to {path}')
    with uproot.recreate(path) as f:
        f['DecayTree'] = tree_dict
    
    print("Added weights")