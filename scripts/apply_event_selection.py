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
from tqdm import tqdm
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
        plt.xlabel('$m(K^+K^-)$ [MeV/$c^2$]')
        plt.ylabel('Frequency')
        plt.title('Phi Mass Distribution')
        plt.legend()
        if save_plot:
            plt.savefig('/ceph/users/togasa/collected_pdfs/selections/phi_mass_distribution.png')

    return df_kin['mask']


def apply_classical_selection(df, features, decay_type, save_plot = False):
    if decay_type == 'Bs2DsPi':
        num_events_before = df["event_entry"].nunique()
        df = df.loc[df['piplus_PID_K'] < 0]
        print(f"PIDK cut efficiency: {df["event_entry"].nunique()/num_events_before:.3f}")

        num_events_before = df["event_entry"].nunique()
        df = df.loc[df['piminus_PID_P'] < 10]
        print(f"PIDP cut efficiency: {df["event_entry"].nunique()/num_events_before:.3f}")

        num_events_before = df["event_entry"].nunique()
        df = df[get_phi_mask(df[features], save_plot=save_plot)]
        print(f"Phi mass cut efficiency: {df["event_entry"].nunique()/num_events_before:.3f}")

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
    parser.add_argument('--batch_size', help='Size of the data batch to process at a time', type=int, default=250_000) 

    cfg = parser.parse_args()
    pprint(cfg)
    os.makedirs(os.path.dirname(cfg.target), exist_ok=True)

    with open(cfg.BDT, 'rb') as f:
        loaded_data = pickle.load(f)
        BDT = loaded_data['model']
        cut = loaded_data['cut']
        
        if cfg.decay_type =='Bs2JpsiKst':
           cut = 0.9 # use a more aggressive cut for the rarer Bs2JpsiKst decay
        

    if cfg.decay_type =='Bs2JpsiKst':
        load_decay = 'Bd2JpsiKst' # Bs2JpsiKst uses same event selection as Bd2JpsiKst
    else:
        load_decay = cfg.decay_type
    
    with open(cfg.signal_class_features, 'r') as f:
        signal_class_features = yaml.safe_load(f)[load_decay]

    with open(cfg.classical_selection_features, 'r') as f:
        classical_selection_features = yaml.safe_load(f)[load_decay]


    with uproot.recreate(cfg.target.replace('.root', '_temp.root')) as fout:
        chunk_iter = uproot.iterate({cfg.data: cfg.treename}, library="pd", step_size=cfg.batch_size)

        for i, chunk in tqdm(enumerate(chunk_iter)):
            chunk["event_entry"] = chunk["file_id"].astype(str) + "_" + chunk["RUNNUMBER"].astype(str) + "_" + chunk["EVENTNUMBER"].astype(str)
            print(f"Chunk file contains {chunk['event_entry'].nunique()} unique events", flush=True)

            chunk = apply_classical_selection(chunk, classical_selection_features, cfg.decay_type)

            # Apply the BDT to select signal events
            chunk_events = chunk[signal_class_features + ['event_entry']].groupby('event_entry').first().reset_index(drop=False)
            chunk_events['signalness'] = BDT.predict_proba(chunk_events[signal_class_features].to_numpy(), chunk_events["event_entry"].values)
            chunk_events = chunk_events[chunk_events['signalness'] > cut]

            #Merge the selected events back to the dataframe, dropping all non-selected events, 
            # and keeping only one candidate per event, which is the one with the highest signalness
            chunk = chunk.merge(chunk_events[["event_entry", "signalness"]], on="event_entry", how="inner")
            del chunk_events

            print(f"Number of events after selection: {chunk['event_entry'].nunique()}", flush=True)

            #If the binning is provided, split the data into bins
            if cfg.binning is not None:
                print(f"Splitting the data into bins according to {cfg.binning}", flush=True)
                binning = cfg.binning.split("of")
                bin_idx = int(binning[0][-1])
                n_bins = int(binning[1])
                binning_var = bin_var_translation[binning[0][:-1]]
                bin_edges = load_bins(cfg.decay_type, binning[0][:-1], cfg.bin_file)

                for a,b in zip(bin_edges[1:], bin_edges[:-1]): #Check binning
                    chunk_bin = chunk[chunk[binning_var].between(float(b), float(a))].groupby("event_entry").first().reset_index(drop=False)
                    print(f"{len(chunk_bin)}", flush=True)

                chunk_events = chunk[['event_entry', binning_var]].groupby("event_entry").first().reset_index(drop=False)
                chunk_events = chunk_events[chunk_events[binning_var].between(float(bin_edges[bin_idx-1]), float(bin_edges[bin_idx]))]

                chunk = chunk.merge(chunk_events[["event_entry"]], on="event_entry", how="inner")
                del chunk_events

            chunk.drop(columns=['event_entry'], inplace=True)

            if i == 0:
                fout["DecayTree"] = chunk
            else:
                fout["DecayTree"].extend(chunk)

            del chunk

    
    # Prevents the case where the file is not fully written and the target file is created, but empty / partially written
    # and snakemake finds a target file and thinks the rule is completed, while it is not
    os.rename(cfg.target.replace('.root', '_temp.root'), cfg.target)
    print(f'Script finished successfully at {datetime.datetime.now().strftime("%H:%M:%S")}', flush = True)