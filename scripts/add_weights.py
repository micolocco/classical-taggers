import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import zfit
import uproot
import pandas as pd
from os.path import join
import mplhep as hep
hep.style.use("LHCb2")
import argparse
from hepstats.splot import compute_sweights
import os
import json

# from zfit.models.physics import DoubleCB
# from zfit.models.functor import SumPDF
# from hepstats.splot import compute_sweights



from scripts.adding_features_v2 import get_loading_vars

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
    # parser.add_argument('--data_fit_model', help='The saved Model of the data massfits', type=str)
    # parser.add_argument('--sim_fit_model', help='The saved Model of the simulation massfits', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--decayType', help='Event decay', type=str)
    parser.add_argument('--obs_name', help='Name of observable', type=str, default="B_DTF_PV_Jpsi_MASS")
    parser.add_argument('--range', help='Observable range', nargs="+")
    parser.add_argument('--out_path', help='Where fit results and plots will be stored', type=str)
    parser.add_argument('--weight_file', help='Root file storing the weights', type=str)



    cfg = parser.parse_args()

    mass_range = (int(cfg.range[0]), int(cfg.range[1]))

    loading_variables_withPrefix = get_loading_vars(cfg.decayType, True)

    #Load Dataset
    with uproot.open(cfg.data_file) as _f:
        df_data = _f[cfg.treename].arrays(loading_variables_withPrefix, library="pd")
    df_data.dropna(inplace=True)
    filenumber = os.path.basename(cfg.data_file)[:-5]
    print(filenumber)
    print(filenumber[10:-9])
    filenumber = int(filenumber[10:-9])


    df_data["event_entry"] =  str(filenumber) + "_" + df_data["RUNNUMBER"].astype(str) + "_" + df_data["EVENTNUMBER"].astype(str)

    df_data = df_data.query(f'{cfg.obs_name} < {mass_range[1]} and {cfg.obs_name} > {mass_range[0]}')

    #Load Weightfile
    with uproot.open(cfg.weight_file) as _f:
        df_weight = _f['DecayTree;1'].arrays(library="pd")
    df_weight.dropna(inplace=True)
    df_weight["event_entry"] =  df_weight["SAMPLENUMBER"].astype(str) + "_" + df_weight["RUNNUMBER"].astype(str) + "_" + df_weight["EVENTNUMBER"].astype(str)

    df_data = df_data.reset_index()
    print(df_data.head())
    print(df_data.shape)
    df_data = df_data[df_data["event_entry"].isin(df_weight["event_entry"])] #Drops all events that have not been selected in the mass_fit script



    df_weight.drop(columns=["B_DTF_PV_Jpsi_MASS", 'RUNNUMBER', 'EVENTNUMBER', "B_ID"], inplace=True)
    df_data = df_data.merge(df_weight.reset_index(), on=["event_entry"], how='left')
    print(df_data.columns)
   
    print(df_data.head())
    print(df_data.shape)

    df_data.drop(columns=['event_entry'], inplace=True)

    #Save weighted dataframe to disk
    tree_dict = {col: np.array(df_data[col]) for col in df_data.columns}

    for col in tree_dict:
        print(f'{col}: {tree_dict[col][0]} ({type(tree_dict[col][0])})')

    path = join(cfg.out_path, os.path.basename(cfg.data_file))
    print(f'writing to {path}')
    with uproot.recreate(path) as f:
        f['DecayTree'] = tree_dict
    
    print("Added weights")