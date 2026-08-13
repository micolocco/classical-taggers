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

import matplotlib.pyplot as plt

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


def get_phi_mask(df, save_plot=False):
    # Calculate the invariant mass of the Phi meson from its decay products (K+ and K-)


    df_kin = df[['hplus_PX', 'hplus_PY', 'hplus_PZ', 'hminus_PX', 'hminus_PY', 'hminus_PZ']]

    m_K = 493.677  # Mass of Kaon in MeV/c^2
    df_kin['E_Kplus'] = np.sqrt(df_kin['hplus_PX']**2 + df_kin['hplus_PY']**2 + df_kin['hplus_PZ']**2 + m_K**2)
    df_kin['E_Kminus'] = np.sqrt(df_kin['hminus_PX']**2 + df_kin['hminus_PY']**2 + df_kin['hminus_PZ']**2 + m_K**2)

    
    df_kin['inv_mass'] = np.sqrt(
                       (df_kin['E_Kplus']  + df_kin['E_Kminus'])**2  -
                      ((df_kin['hplus_PX'] + df_kin['hminus_PX'])**2 + 
                       (df_kin['hplus_PY'] + df_kin['hminus_PY'])**2 + 
                       (df_kin['hplus_PZ'] + df_kin['hminus_PZ'])**2))

    if save_plot:
        mass_sorted = np.sort(df_kin['inv_mass'])
        m_min = mass_sorted[int(len(mass_sorted)*0.01)]
        m_max = mass_sorted[int(len(mass_sorted)*0.99)]
        bins = np.linspace(m_min, m_max, 100)
        plt.hist(df_kin['inv_mass'], bins=bins, alpha=0.7, color='blue', label='Before phi mass cut')

    # Constrain the inv mass of the intermediate phi to [1000, 1040] MeV/c^2
    df_kin['mask'] = (df_kin['inv_mass'] >= 1000) & (df_kin['inv_mass'] <= 1040)

    if save_plot:
        plt.hist(df_kin.loc[df_kin['mask'], 'inv_mass'], bins=bins, alpha=0.7, color='orange', label='After phi mass cut')
        plt.xlabel('$m(K^+K^-)$ [MeV/c^2]')
        plt.ylabel('Frequency')
        plt.title('Phi Mass Distribution')
        plt.legend()
        if save_plot:
            plt.savefig('/ceph/users/togasa/collected_pdfs/selections/phi_mass_distribution.png')

    return df_kin['mask']


def apply_classical_selection(df, features, decay_type, save_plot = False):
    if decay_type == 'Bs2DsPi':
        num_events_before = len(df)
        df = df.loc[df['piplus_PID_K'] < 0]
        print(f"PIDK cut efficiency: {len(df)/num_events_before:.3f}")

        num_events_before = len(df)
        df = df.loc[df['piminus_PID_P'] < 10]
        print(f"PIDP cut efficiency: {len(df)/num_events_before:.3f}")

        num_events_before = len(df)
        df = df[get_phi_mask(df[features], save_plot=save_plot)]
        print(f"Phi mass cut efficiency: {len(df)/num_events_before:.3f}")

        return df
    else:
        return df
    

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
    parser.add_argument('--classical_selection_features', help='Path to the yaml file containing the features used for classical selection')
    parser.add_argument('--bin_file', help='Path to the file containing the bin edges for the variable used for binning the data')

    cfg = parser.parse_args()
    pprint(cfg)
    os.makedirs(os.path.dirname(cfg.target), exist_ok=True)

    with open(cfg.BDT, 'rb') as f:
        loaded_data = pickle.load(f)
        BDT = loaded_data['model']
        cut = loaded_data['cut']
        
        if cfg.decay_type =='Bs2JpsiKst':
           cut = 0.9 # use a more aggressive cut for the rarer Bs2JpsiKst decay
        
    with uproot.open(cfg.data) as f:
        df = f[cfg.treename].arrays(library='pd')

    if cfg.decay_type =='Bs2JpsiKst':
        load_decay = 'Bd2JpsiKst' # Bs2JpsiKst uses same event selection as Bd2JpsiKst
    else:
        load_decay = cfg.decay_type
    
    with open(cfg.signal_class_features, 'r') as f:
        signal_class_features = yaml.safe_load(f)[load_decay]

    with open(cfg.classical_selection_features, 'r') as f:
        classical_selection_features = yaml.safe_load(f)[load_decay]

    df["event_entry"] = df["file_id"].astype(str) + "_" + df["RUNNUMBER"].astype(str) + "_" + df["EVENTNUMBER"].astype(str)
    df['candidate_entry'] = df['file_id'].astype(str) + "_" + df['candidate_index'].astype(str)
    print(f"Input file contains {df['event_entry'].nunique()} unique events", flush=True)

    df = apply_classical_selection(df, classical_selection_features, cfg.decay_type, save_plot = '00001_1' in  os.path.basename(cfg.target)) #only save the plot for the first file 

    # Apply the BDT to select signal events
    # Grouped by candidate_entry not event_entry because one event may have several candidates, due to (almost purely) incorrect reconstruction, 
    # multiplicity is removed after prediction
    df_candidates = df[signal_class_features + ['event_entry', 'candidate_entry']].groupby('candidate_entry').first().reset_index(drop=False)
    print(f"Prediction of signalness starts at {datetime.datetime.now()}", flush=True)
    print(signal_class_features, flush=True)
    print(df_candidates[signal_class_features].head(), flush=True)
    df_candidates['signalness'] = BDT.predict_proba(df_candidates[signal_class_features].to_numpy(), df_candidates["candidate_entry"].values)
    print(f"Prediction of signalness ends at {datetime.datetime.now()}", flush=True)
    df_candidates = df_candidates[df_candidates['signalness'] > cut]

    #Drop Multiplicit candidates, i.e. events with more than one candidate passing the selection, almost allways incorrect reconstructions
    #-> Choose one candidate per event, which is the one with the highest signalness, as the most probable correct reconstruction
    df_candidates = df_candidates.sort_values('signalness').groupby("event_entry").last().reset_index(drop=False)

    #Merge the selected candidates back to the dataframe, dropping all non-selected candidates, 
    # and keeping only one candidate per event, which is the one with the highest signalness

    print(list(df.columns), flush=True)
    print(list(df_candidates.columns), flush=True)

    df = df.merge(df_candidates[["candidate_entry", "signalness"]], on="candidate_entry", how="inner")
    del df_candidates

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


