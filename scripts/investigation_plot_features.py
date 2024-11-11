import numpy as np 
import pandas as pd 
from sklearn import tree
import sys 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import graphviz 
from sklearn.metrics import accuracy_score, roc_curve ,auc
import time
import uproot
import os
import glob
import argparse
# Local import
from IPython import embed
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
plt.rcParams['text.usetex'] = False # HD cluster has some problems with dvp not found
plt.rcParams.update({'axes.unicode_minus' : False})
import DT_utils

'''
Origin Flag IDs:

0 == Signal
1 == SS Fragmentation
2 == OS Decay (track has B0, B+, Bs, Bc+ mother)
3 == OS Fragmentationfrom excited B
4 == OS Frag from b quark
5 == Prompt
100 == Tracks from other vertex
-1 == No associated MC particle (probably ghost)

'''


def find_tree_name(decay):
    if decay == 'Bs2JpsiPhi':
        return 'BsToJpsiPhi_Detached/DecayTree'
    #if decay == 'Bs2DsPi':
    #    return 'Hlt2B2OC_BdToDsmPi_DsmToKpKmPim/DecayTree' # For file of type root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000001_1.mc.root
    else:
        return 'Tuple/DecayTree'

def plot_features(data, features, nbins=100):
    data = data[data['particle']=='SSPion']
    keys_list = pd.unique(data['particle']).tolist()
    print(f'Particle: {keys_list}')
    # Plot input features 
    for particle in keys_list:
        plt.figure(figsize=(100,100))
        for i, col in enumerate(features):
            plt.subplot(10, 5, i + 1)
            plt.xlabel(col)
            #plt.hist(data[col][(data['particle']==particle) & (abs(data['B_Tr_T_MC_MOTHER_ID'])==5) & (data['sign_tag']==1) & (data['B_Tr_T_Charge']==1)], density = True, bins=nbins, label = f"sign=1", histtype='step', color='m', lw=2)
            #plt.hist(data[col][(data['particle']==particle) & (abs(data['B_Tr_T_MC_MOTHER_ID'])==5) & (data['sign_tag']==-1) &  (data['B_Tr_T_Charge']==1)], density = True, bins=nbins, label = f"sign=-1", histtype='step', color='b', lw=2)  
            #plt.hist(data[col][(data['particle']==particle) & (data['sign_tag']==1) ], density = True, bins=nbins, label = f"sign=1", histtype='step', color='m', lw=2)
            #plt.hist(data[col][(data['particle']==particle) & (data['sign_tag']==-1) ], density = True, bins=nbins, label = f"sign=-1", histtype='step', color='b', lw=2)  
            #plt.legend() 
            #plt.yscale('log')
            data = data[(data.B_Tr_T_MC_MOTHER_ID==5)|(data.B_Tr_T_MC_MOTHER_ID==-5)]
            # Define the number of bins
            nbins = 50

            # Get histogram data for each subset
            hist_sign1, bins = np.histogram(data[col][(data['particle']==particle) & (data['sign_tag']==1)], 
                                            bins=nbins, density=True)

            hist_sign_minus1, _ = np.histogram(data[col][(data['particle']==particle) & (data['sign_tag']==-1)], 
                                               bins=bins, density=True)

            # Calculate the bin centers for plotting
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Compute the difference between the histograms
            hist_diff = hist_sign1 - hist_sign_minus1

            # Plot the two histograms
            plt.hist(data[col][(data['particle']==particle) & (data['sign_tag']==1)], 
                     density=True, bins=bins, label="sign=1", histtype='step', color='m', lw=2)

            plt.hist(data[col][(data['particle']==particle) & (data['sign_tag']==-1)], 
                     density=True, bins=bins, label="sign=-1", histtype='step', color='b', lw=2)

            # Plot the difference
            plt.step(bin_centers, hist_diff, where='mid', label="Difference (sign=1 - sign=-1)", color='r', lw=2)

            # Add labels and legend
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()

        plt.tight_layout()
        plt.title(f'{particle}')
        plt.savefig(f"{cfg.target_path}/DT_features_{particle}_diff5.pdf")
        print(f"Saved plot {cfg.target_path}/DT_features_{particle}.pdf")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='/ceph/users/molocco/Data/withUT_MC_2024/DT_outputs/origin_investigations')    
    cfg = parser.parse_args()

    from pprint import pprint   
    pprint(cfg)
    start = time.time()

    run_time = time.time()
    # Check and eventually make output directory where training info will be saved
    os.makedirs(cfg.target_path, exist_ok=True)

    base_pattern = '/ceph/users/molocco/Data/withUT_MC_2024/2_added_features'

    # set1 --> only uses PIDs and IP significance of the B primary vertex
    # set2 --> collects the pre-selections features used in run2 (except SSKaon). See https://gitlab.cern.ch/lhcb/Phys/-/tree/run2-patches/Phys/FlavourTagging/python/FlavourTagging
    features_set1 = ['B_Tr_T_PIDK', 'B_Tr_T_PIDe', 'B_Tr_T_PIDmu', 'B_Tr_T_PIDP', 'B_Tr_T_BVIPSig'] 
    features_set2 = ['B_Tr_T_P' , 'B_Tr_T_TRACKISLONG', 'B_Tr_T_CHI2DOF', 'B_Tr_T_minPhiDistance', 'B_Tr_T_ISMUON', 'B_Tr_T_GHOSTPROB', 'B_Tr_T_absIP', \
                    'B_Tr_T_eoverP', 'B_Tr_T_BPVIPCHI2', 'B_Tr_T_PT', 'B_Tr_T_DeltaQ_Pion', 'B_Tr_T_DeltaQ_Muon', 'B_Tr_T_DeltaQ_Electron', 'B_Tr_T_DeltaQ_Proton', 'B_Tr_T_DeltaQ_Kaon', 'B_Tr_T_Signal_TagPart_PT', 'B_Tr_T_EtaDistance', 'B_Tr_T_PhiDistance', 'B_Tr_T_DeltaR', 'B_Tr_T_Charge'] 
    # set3 --> on the top of set1 and set2 adds other variables from https://gitlab.cern.ch/lhcb/Phys/-/blob/run2-patches/Phys/FlavourTagging/python/FlavourTagging/DevelopmentTaggerConf.py
    features_set3 = ['B_Tr_T_cos_PhiDistance', 'B_Tr_T_diff_z', 'B_Tr_T_PX', 'B_Tr_T_PY', 'B_Tr_T_PZ', 'B_Tr_T_ENERGY', 'B_Tr_T_Eta', 'B_Tr_T_Phi', 'B_Tr_T_BPVIP', 'B_PT', 'B_nTracks', 'B_Tr_T_MINIP', 'B_Tr_T_MINIPChi2', 'B_nPVs', 'B_Tr_T_atanPT_PZ'] 
    features_set4 = ['B_Tr_T_BPVX', 'B_Tr_T_BPVY','B_Tr_T_BPVZ',]
    # Missing PROBNN for all the particles (not usable yet)
    # atan ((PT/PZ)) possible to implement as function
    # TRPCHI2 dropped (CHI2 probability)
    # CLONEDIST could be replaced with TRACKISCLONE functor. Not in our Analysis Production (AP)
    # PP_InAccHcal could be replaced with INHCAL. Not in our Analysis Production (AP)
    # PP_VeloCharge not available functor. It can be replaced with HASVELO. Not in our Analysis Production (AP)
    # IPPUSig not available functor
    # Signal_TagPart_CHI2DOF not available functor
    # TRLH = track likelihood. Dropped
    # SumBDT_ult don't know what is
    # PVndof not clear
    # TRGHP alias for TRACKGHOSTPROB
    features = features_set1+features_set2+features_set3
    loading_variables = features +["B_Tr_T_absID", "B_Tr_T_Origin_Flag", "B_TRUEID", "B_Tr_T_MC_MOTHER_ID"]
    # Path to input root files
    #file_pattern = f'/ceph/users/molocco/classical-taggers/Data/{config.sample_type}/2_added_features/*/*.root'
    # Define the base pattern and the folders of interest
    #base_pattern = '/ceph/users/molocco/Data/withUT_MC_2024/2_added_features'

    
    #folders = ['Bd2JpsiKst', 'Bs2DsPi', 'Bu2JpsiK']
    folders = ['Bd2JpsiKst',] # Curiouys to see with PROBNNs
    # NEED TO INCLUDE BS!!!!!!!


    # Initialize a list to store the paths of root files
    
    
    # Iterate over each folder and collect the root files
    print(f"Loading data: Start \n")

    df = pd.DataFrame(columns=loading_variables)
    
    for decay in folders:
        pattern = f'{base_pattern}/{decay}/*.root'
        root_files = []
        root_files.extend(glob.glob(pattern))
        treename = find_tree_name(decay)
        for f in root_files:
            print(f"Reading input file: {f}")
            with uproot.open("{}".format(f)) as _f:
                _df = _f[treename].arrays(loading_variables, library="pd")
                _df['decay'] = decay
                df = pd.concat([df, _df], ignore_index = True)
    print(df.shape[0]) 
    
    # For SS case if B_Tr_T_Charge has same sign of B_TRUE_ID is a correct tagging particle candidate
    # We want to remove all the SSKaon from Bd2JpsiKst and the SSPion, SSProton from Bs2DsPi
    # We want to remove all the SSKaon (or SSPion/SSProton) that will return a wrong tagging decision
    '''
    
    # For test    
    root_files=[
            '/ceph/users/molocco/Data/withUT_MC_2024/2_added_features/Bd2JpsiKst/00214047_00000001_1.mc.root',]
   
    for f in root_files:
        print(f"Reading input file: {f}")
        with uproot.open("{}".format(f)) as _f:
            _df = _f['Tuple/DecayTree'].arrays(loading_variables, library="pd")
            _df['decay'] = 'Bd2JpsiKst'
        df = pd.concat([df, _df], ignore_index = True)
    
    
    print(f"Loading data finished in {round(-start+ time.time() , 2)}s")
    '''
    

    df.B_Tr_T_Origin_Flag.astype(int)
    df[['B_Tr_T_MC_MOTHER_ID']].astype(int)
    # Define labels for multiclassification
    conditions = [
    (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==2), # OSKaon
    (df.B_Tr_T_absID==13) & (df.B_Tr_T_Origin_Flag==2), # OSMuon
    (df.B_Tr_T_absID==11) & (df.B_Tr_T_Origin_Flag==2) & (abs(df.B_Tr_T_MC_MOTHER_ID)!=22), # OSElectron, remove OSElectron from photon splitting
    (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1), # SSPion
   # ((df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1)) | ((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1)), # SSProton and SSKaon
    (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1), # SSProton
    (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1), # SSKaon
    (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==2), # OSProton
    (df.B_Tr_T_Origin_Flag==100)
    ]
    particle_type = {"OSKaon":1,
                    "OSMuon":2,
                    "OSElectron":3,
                    "SSPion":4,
                    #"SSProton+SSKaon": 5,
                    "SSProton":5,
                    "SSKaon":6,
                    "OSProton":7,
                    "notSamePV":8 }
    df['ID_type'] = np.select(conditions, particle_type.values())
    df.loc[~df['ID_type'].isin(particle_type.values()), 'ID_type'] = 0
    # Assign the corresponding particle type
    df['particle'] = df['ID_type'].map({v: k for k, v in particle_type.items()})
    # If ID_type is not in particle_type values, set 'particle' to None
    df.loc[~df['ID_type'].isin(particle_type.values()), 'particle'] = 'not_taggingPart'
    df = df.loc[(df.ID_type != 0 )]
    # For SS case if B_Tr_T_Charge has same sign of B_TRUE_ID is a correct tagging particle candidate
    # We want to remove all the SSKaon from Bd2JpsiKst and the SSPion, SSProton from Bs2DsPi
    # We want to remove all the SSKaon (or SSPion/SSProton) that will return a wrong tagging decision
    df['sign_tag'] = (df['B_TRUEID']/abs(df['B_TRUEID'])) * df['B_Tr_T_Charge']
    print(df['B_TRUEID'].value_counts())
    plot_features(df, features, )