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
Ref: https://gitlab.cern.ch/lhcb/Rec/-/blob/master/Phys/DaVinciMCKernel/src/Lib/MCTaggingHelper.cpp?ref_type=heads
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

def plot_features_byOrigin(data, features, particle_type, nbins=100):
    # Plot input features 
    plt.figure(figsize=(100,100))
    for i, col in enumerate(features):
        plt.subplot(10, 8, i + 1)
        # Ranges and names must be adapted
        #plt.hist(data[col][data['particle']==particle_type['OSKaon']], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=)
        #plt.hist(data[col][data['particle']==particle_type['OSMuon']], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=ranges[col])
        #plt.hist(data[col][data['particle']==particle_type['OSElectron']], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=ranges[col])
        #plt.hist(data[col][data['particle']==particle_type['SSPion']], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=ranges[col])
        #plt.hist(data[col][data['particle']==particle_type['SSProton']], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=ranges[col])
        #plt.hist(data[col][data['particle']==particle_type['SSKaon']], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=ranges[col])
       

        #plt.hist(data[col][data['particle']==6], density = True, bins=nbins, label = f"{particle_type[6]}", histtype='step', color='r', )
        if col in nice_names.keys():
            plt.xlabel(nice_names[col])
            plt.hist(data[col][data['particle']=='OSKaon'], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='OSMuon'], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='OSElectron'], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='SSPion'], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='SSProton'], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=ranges[col])
            #plt.hist(data[col][data['particle']=='SSKaon'], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=ranges[col])
        else:
            plt.xlabel(col)
            plt.hist(data[col][data['particle']=='OSKaon'], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, )
            plt.hist(data[col][data['particle']=='OSMuon'], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, )
            plt.hist(data[col][data['particle']=='OSElectron'], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, )
            plt.hist(data[col][data['particle']=='SSPion'], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, )
            plt.hist(data[col][data['particle']=='SSProton'], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, )
            #plt.hist(data[col][data['particle']=='SSKaon'], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, )
            plt.hist(data[col][data['particle']=='OSProton'], density = True, bins=nbins, label = f"OSProton", histtype='step', color='orange', lw=2, )
        plt.legend() 
        plt.tight_layout()
    plt.savefig(f"{cfg.target_path}/DT_features_byOrigin.pdf")

def plot_features_byParticle(data, features, nbins=100):
   
    plt.figure(figsize=(100,100))
    for i, col in enumerate(features):
        plt.subplot(10, 8, i + 1)
        #plt.hist(data[col][data.B_Tr_T_absID==321], density = True, bins=nbins, label = f"Kaon", histtype='step', color='m', lw=2, range=ranges[col])
        #plt.hist(data[col][data.B_Tr_T_absID==13], density = True, bins=nbins, label = f"Muon", histtype='step', color='b', lw=2, range=ranges[col])
        #plt.hist(data[col][data.B_Tr_T_absID==11], density = True, bins=nbins, label = f"Electron", histtype='step', color='c', lw=2, range=ranges[col])
        #plt.hist(data[col][data.B_Tr_T_absID==211], density = True, bins=nbins, label = f"Pion", histtype='step', color='g', lw=2, range=ranges[col])
        #plt.hist(data[col][data.B_Tr_T_absID==2212], density = True, bins=nbins, label = f"Proton", histtype='step', color='y', lw=2, range=ranges[col])
        plt.hist(data[col][data.B_Tr_T_absID==321], density = True, bins=nbins, label = f"Kaon", histtype='step', color='m', lw=2, )
        plt.hist(data[col][data.B_Tr_T_absID==13], density = True, bins=nbins, label = f"Muon", histtype='step', color='b', lw=2, )
        plt.hist(data[col][data.B_Tr_T_absID==11], density = True, bins=nbins, label = f"Electron", histtype='step', color='c', lw=2, )
        plt.hist(data[col][data.B_Tr_T_absID==211], density = True, bins=nbins, label = f"Pion", histtype='step', color='g', lw=2, )
        plt.hist(data[col][data.B_Tr_T_absID==2212], density = True, bins=nbins, label = f"Proton", histtype='step', color='y', lw=2, )
        #plt.hist(data[col][data['particle']==6], density = True, bins=nbins, label = f"{particle_type[6]}", histtype='step', color='r', )
        if col in nice_names.keys():
            plt.xlabel(nice_names[col])
        else:
            plt.xlabel(col)
        plt.legend() 
        plt.tight_layout()
    plt.savefig(f"{cfg.target_path}/DT_features_byParticle.pdf")

def find_tree_name(decay):
    if decay == 'Bs2JpsiPhi':
        return 'BsToJpsiPhi_Detached/DecayTree'
    #if decay == 'Bs2DsPi':
    #    return 'Hlt2B2OC_BdToDsmPi_DsmToKpKmPim/DecayTree' # For file of type root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000001_1.mc.root
    else:
        return 'Tuple/DecayTree'

def plot_features_bySign(data, features, nbins=100):
    keys_list = pd.unique(data['particle']).tolist()
    print(f'Particle: {keys_list}')
    # Plot input features 
    for particle in keys_list:
        plt.figure(figsize=(100,100))
        for i, col in enumerate(features):
            plt.subplot(10, 8, i + 1)
            plt.xlabel(col)
            #plt.hist(data[col][(data['particle']==particle) & (abs(data['B_Tr_T_MC_MOTHER_ID'])==5) & (data['sign_tag']==1) & (data['B_Tr_T_Charge']==1)], density = True, bins=nbins, label = f"sign=1", histtype='step', color='m', lw=2)
            #plt.hist(data[col][(data['particle']==particle) & (abs(data['B_Tr_T_MC_MOTHER_ID'])==5) & (data['sign_tag']==-1) &  (data['B_Tr_T_Charge']==1)], density = True, bins=nbins, label = f"sign=-1", histtype='step', color='b', lw=2)  
            plt.hist(data[col][(data['particle']==particle) & (data['sign_tag']==1) ], density = True, bins=nbins, label = f"sign=1", histtype='step', color='m', lw=2,)
            plt.hist(data[col][(data['particle']==particle) & (data['sign_tag']==-1) ], density = True, bins=nbins, label = f"sign=-1", histtype='step', color='b', lw=2)  
            plt.legend() 
            #plt.yscale('log')
        plt.tight_layout()
        plt.title(f'{particle}')
        plt.savefig(f"{cfg.target_path}/DT_features_{particle}.pdf")
        print(f"Saved plot {cfg.target_path}/DT_features_{particle}.pdf")

def count_BKGCAT(df):
    # Assuming you have a specific particle type in mind, like "OSElectron"
    particle_list = pd.unique(df['particle']).tolist()
    for particle in particle_list:
        # Filter the DataFrame for the specified particle
        df_particle = df[df['particle'] == particle]
        # Calculate the percentage of each B_BKGCAT value
        bkgcat_counts = df_particle['B_BKGCAT'].value_counts(normalize=True) * 100
        # Print the results
        print(f"Percentage distribution of B_BKGCAT values for {particle}:")
        for bkgcat, percentage in bkgcat_counts.items():
            print(f"B_BKGCAT {bkgcat}: {percentage:.2f}%")
        print('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    #base_pattern = '/ceph/users/molocco/Data/withUT_MC_2024/2_added_features'
    parser.add_argument('--base_pattern', help='Pattern for input files', type=str)
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='/ceph/users/molocco/Data/withUT_MC_2024/DT_outputs')
    parser.add_argument('--balanced', help='If classes are balanced or unbalanced', choices=('balanced', 'unbalanced'), type=str, )
    parser.add_argument('--unify_SS', help='If unify SSKaon and SSProton in a single class', action='store_true' ) # action='store_true' means args.unify_SS will be set to True if the --unify_SS argument is provided on the command line.
    parser.add_argument('--BKG0', help='If specified, only BGKCAT=0 tracks are used',  action='store_true') # action='store_true' means args.BKG0 will be set to True if the --BKG0 argument is provided on the command line.

    cfg = parser.parse_args()

    from pprint import pprint   
    pprint(cfg)
    start = time.time()
    # Check and eventually make output directory where training info will be saved
    os.makedirs(cfg.target_path, exist_ok=True)
    fetaures_added = ['B_Tr_T_minPhiDistance', 'B_Tr_T_cos_PhiDistance', 'B_Tr_T_PhiDistance', 'B_Tr_T_diff_z', 'B_Tr_T_DeltaR', 'diff_P', 'P_proj', 't', 'EVIP', 'B_Tr_T_absBPVIP', 'B_Tr_T_EtaDistance', 'B_Tr_T_DeltaQ_Pion', 'B_Tr_T_DeltaQ_Muon', 'B_Tr_T_DeltaQ_Electron', 'B_Tr_T_DeltaQ_Proton', 'B_Tr_T_DeltaQ_Kaon', 'B_Tr_T_Signal_TagPart_PT', 'B_Tr_T_eoverP', 'B_Tr_T_BPVIPSig', 'logEVIP', 'logP_proj', 'B_Tr_T_atanPT_PZ']
    loading_variables_noMC = [
        'B_BPVX',
        'B_BPVY',
        'B_BPVZ',
        'B_END_VX',
        'B_END_VY',
        'B_END_VZ',
        'B_ENERGY',
        'B_ETA',
        'B_M',
        'B_P',
        'B_PHI',
        'B_PT',
        'B_PX',
        'B_PY',
        'B_PZ',
        'B_nPVs',
        'B_nTracks',
        'B_Tr_T_TRACKISLONG',
        'B_Tr_T_OWNPVIP',
        'B_Tr_T_OWNPVIPCHI2',
        'B_Tr_T_BPVIP',
        'B_Tr_T_BPVIPCHI2',
        'B_Tr_T_Charge',
        'B_Tr_T_ISMUON',
        'B_Tr_T_ENERGY',
        'B_Tr_T_Eta',
        'B_Tr_T_MINIP',
        'B_Tr_T_MINIPChi2',
        'B_Tr_T_P',
        'B_Tr_T_PT',
        'B_Tr_T_PIDK',
        'B_Tr_T_PIDe',
        'B_Tr_T_PIDmu',
        'B_Tr_T_PIDP',
        'B_Tr_T_PROBNN_GHOST',
        'B_Tr_T_PROBNN_E',
        'B_Tr_T_PROBNN_K',
        'B_Tr_T_PROBNN_P',
        'B_Tr_T_PROBNN_MU',
        'B_Tr_T_PROBNN_PI',
        'B_Tr_T_zfirst', # This is the reconstructed z coordinate where a track begins (basically the PV of a track, similar to B_OWNPV_Z but for a track)
        'B_Tr_T_BPVX',
        'B_Tr_T_BPVY',
        'B_Tr_T_BPVZ',
        'B_Tr_T_Phi',
        'B_Tr_T_M',
        'B_Tr_T_CHI2DOF',
        'B_Tr_T_GHOSTPROB',
        'B_Tr_T_PX',
        'B_Tr_T_PY',
        'B_Tr_T_PZ',
        'B_Tr_T_X',
        'B_Tr_T_Y',
        'B_Tr_T_Z',
         ]
        
    # Missing fetaures wrt Run2
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
    features = loading_variables_noMC + fetaures_added
    loading_variables = features +["B_BKGCAT", "B_Tr_T_absID", "B_Tr_T_Origin_Flag", "B_TRUEID", "B_Tr_T_MC_MOTHER_ID"]    
 
    #folders = ['Bd2JpsiKst', 'Bs2DsPi', 'Bu2JpsiK']
    folders = ['Bd2JpsiKst',  'Bu2JpsiK'] 
    # NEED TO INCLUDE BS!!!!!!!

    # Iterate over each folder and collect the root files
    print(f"Loading data: Start \n")

    df = pd.DataFrame(columns=loading_variables)
    
    for decay in folders:
        pattern = f'{cfg.base_pattern}/{decay}/*5_1.mc.root'
        root_files = []
        root_files.extend(glob.glob(pattern))
        treename = find_tree_name(decay)
        for f in root_files:
            print(f"Reading input file: {f}")
            with uproot.open("{}".format(f)) as _f:
                _df = _f[treename].arrays(loading_variables, library="pd")
                _df['decay'] = decay
                #print(f'Number of tracks per file: {_df.shape[0]}')
                df = pd.concat([df, _df], ignore_index = True)
                #print(f'Number of tracks in concatenated df {df.shape[0]}')

    df.dropna(inplace=True)
    print(f"Total number of tracks: {df.shape[0]}") 

    df.B_Tr_T_Origin_Flag.astype(int)
    df[['B_Tr_T_MC_MOTHER_ID']].astype(int)
    # Define labels for multiclassification
    # List of (condition, particle_type) tuples
    if cfg.unify_SS:
        print("Unifying SSProton and SSKoan classes")
        condition_particle_pairs = [
        ((df.B_Tr_T_absID == 321) & (df.B_Tr_T_Origin_Flag == 2), "OSKaon"),
        ((df.B_Tr_T_absID == 13) & (df.B_Tr_T_Origin_Flag == 2), "OSMuon"),
        ((df.B_Tr_T_absID == 11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) != 22), "OSElectron"),
        ((df.B_Tr_T_absID == 211) & (df.B_Tr_T_Origin_Flag == 1), "SSPion"),
        (((df.B_Tr_T_absID == 2212) | (df.B_Tr_T_absID == 321)) & (df.B_Tr_T_Origin_Flag == 1), "SSProton+SSKaon"), #((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1), "SSKaon"),
        ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 2), "OSProton"),
        ((df.B_Tr_T_Origin_Flag == 100), "notSamePV"),
        ] 

    else:
        condition_particle_pairs = [
        ((df.B_Tr_T_absID == 321) & (df.B_Tr_T_Origin_Flag == 2), "OSKaon"),
        ((df.B_Tr_T_absID == 13) & (df.B_Tr_T_Origin_Flag == 2), "OSMuon"),
        ((df.B_Tr_T_absID == 11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) != 22), "OSElectron"),
        ((df.B_Tr_T_absID == 211) & (df.B_Tr_T_Origin_Flag == 1), "SSPion"),
        ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1), "SSProton"),
        ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 2), "OSProton"),
        ((df.B_Tr_T_Origin_Flag == 100), "notSamePV"),
        ] #((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1), "SSKaon"),

    # Separate conditions and particle types for np.select()
    conditions = [pair[0] for pair in condition_particle_pairs]
    particle_type = [pair[1] for pair in condition_particle_pairs]

    # Assign particle types based on conditions, with default "Others" for unmatched rows
    df['particle'] = np.select(conditions, particle_type, default="Others")
    

    # To remove all the other particles
    # df = df.loc[(df.particle != 0 )]
 
    # For SS case if B_Tr_T_Charge has same sign of B_TRUE_ID is a correct tagging particle candidate
    # We want to remove all the SSKaon from Bd2JpsiKst and the SSPion, SSProton from Bs2DsPi
    # We want to remove all the SSKaon (or SSPion/SSProton) that will return a wrong tagging decision
    df['sign_tag'] = (df['B_TRUEID']/abs(df['B_TRUEID'])) * df['B_Tr_T_Charge']
    print(f"Total number of tracks for each B candidate (by TRUE_ID): \n {df['B_TRUEID'].value_counts()}")
    #plot_features(df, features, )

    removal_conditions = (
        # Remove all SSPions/SSProtons from Bs2DsPi
        (((df['particle']=='SSPion') | (df['particle']=='SSProton')) & (df['decay']=='Bs2DsPi'))
        |  
        # Remove all SSKaons from Bd2JpsiKst
        ((df['particle']=='SSKaon') & (df['decay']=='Bd2JpsiKst'))
        |
        # Remove all SS particles from Bu2JpsiK
        (((df['particle']=='SSKaon') | (df['particle']=='SSPion') | (df['particle']=='SSProton')) & (df['decay']=='Bu2JpsiK'))
        )
        
    '''
    For the moment I want to start without putting conditions on the charge-flavour relation 
    |
    # Remove all OS particles from Bs2DsPi, Bd2JpsiKst
    (((df['particle']=='OSKaon') | (df['particle']=='OSMuon') | (df['particle']=='OSElectron')) & ((df['decay']=='Bd2JpsiKst') | (df['decay']=='Bs2DsPi')))
    |
    # Remove particles that don't give a correct charge-flavour relation
    ((df['particle']=='SSKaon') & (df['sign_tag']==-1) & (df['decay']=='Bs2DsPi'))
    |
    ((df['particle']=='SSPion') & (df['sign_tag']==-1) & (df['decay']=='Bd2JpsiKst'))    
    |
    ((df['particle']=='SSProton') & (df['sign_tag']==+1) & (df['decay']=='Bd2JpsiKst'))
    )
    '''
          
    
    # Filter the DataFrame to remove rows that meet the combined condition
    df_filtered = df[~removal_conditions].copy()
    df_filtered = df_filtered.dropna()

    if cfg.BKG0:
        df_filtered.B_BKGCAT.astype(int)
        print(f"Dropping BKGCAT !=0 tracks...")
        df_filtered = df_filtered[df_filtered.B_BKGCAT==0]
        print(f"New number of tracks: {df_filtered.shape[0]}")
    else:
        count_BKGCAT(df_filtered)
    print(f"\nComposition before downsampling:\n{round(df_filtered.particle.value_counts()/df_filtered.shape[0],4)*100}")
    
    
    # Downsample the 'notSamePV', 'Others' classes. 
    # Get the count of the largest class excluding "notSamePV"
    max_class_size = df_filtered[df_filtered.particle == 'SSPion'].particle.value_counts().max()
    # Filter the 'notSamePV' rows
    not_same_pv_rows = df_filtered[df_filtered.particle == 'notSamePV']
    others_rows = df_filtered[df_filtered.particle == 'Others']
    # Randomly sample the maximum class size from 'notSamePV'
    sampled_not_same_pv = not_same_pv_rows.sample(n=max_class_size, random_state=42)
    others_rows = others_rows.sample(n=max_class_size, random_state=42)
    # Filter out 'notSamePV' from the original dataframe to keep the other rows
    df_filtered = df_filtered[(df_filtered.particle != 'notSamePV') & (df_filtered.particle != 'Others')]
    # Concatenate the sampled 'notSamePV' rows back with the other classes
    df_filtered = pd.concat([df_filtered, sampled_not_same_pv])
    df_filtered = pd.concat([df_filtered, others_rows])
    # Optionally, shuffle the dataframe (to mix rows)
    
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    

    # Plot features
    output_dir = 'DT_outputs'    
    # Unify SSKaon and SSProton into a single class
    #if cfg.unify_SS:
      #  print("Unifying SSProton and SSKoan classes")
        # Unify classes 
      #  df_filtered.loc[(df_filtered.particle=='SSKaon')|(df_filtered.particle=='SSProton'), 'particle']='SSKaon+SSProton'
        # Rescale ID
        #df_filtered.loc[(df_filtered.particle=='SSKaon+SSProton'), 'particle']=5
        #df_filtered.loc[(df_filtered.particle=='OSProton'), 'particle']=6
        #particle_type = {"OSKaon":1,
                       # "OSMuon":2,
                       # "OSElectron":3,
                       # "SSPion":4,
                       # "SSProton+SSKaon": 5,
                       # "OSProton":6,
                       # "notSamePV":7,
                       # "Others": 0,
                       # "prompt": 8
                       # }

    #plot_features_byOrigin(df_filtered, features, particle_type, nbins=50)
    x = df_filtered[features + ["particle"]]

    print(f"\nComposition after downsampling:\n{round(x.particle.value_counts()/x.shape[0],4)*100}")
    #print(f"\nComposition by particle and background category:\n{round(df_filtered[['particle','B_BKGCAT']].value_counts()/df_filtered.shape[0],4)*100}")
    print(f'The features used are {len(features)}: {features}')
    print('-----------------------------------------')
    # To get same amount of not_taggingPart
    #x = pd.concat([x, df.loc[df.particle == 0][features + ["particle"]].head(len(x))])
    y = x.particle
    x.drop(columns=["particle"] , inplace = True)
    x_train , x_test ,y_train, y_test= train_test_split(x, y, test_size = 0.01, random_state=42)
    
    print("Start fitting")
    if cfg.balanced == 'unbalanced':
        weights = None
    else:
        weights = str(cfg.balanced)

    start_fit = time.time()
    clf = tree.DecisionTreeClassifier(max_depth = 6,class_weight=weights, min_impurity_decrease=0.009) 

    clf.fit(x_train, y_train)
    print(f'Decision Tree training required: {round(time.time()-start_fit, 2)}s')
    # Visualize the decision tree
    #dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True,special_characters=True) 
    dot_data = tree.export_graphviz(clf,feature_names=features,class_names=sorted(y_train.unique()),filled=True, rounded=True, special_characters=True, proportion=True) 
    graph = graphviz.Source(dot_data) 
    if cfg.unify_SS:
        title = f'{cfg.balanced}_SSKSSP'
    else:
        title = f'{cfg.balanced}'
    graph.render(f"{cfg.target_path}/{title}_treeSchema")
    
    #print(f"Accuracy:{clf.score(x_test,y_test)}")

    print("\n Metrics for particle type composition: true VS predicted\n")
    DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), possible_particle=sorted(df_filtered['particle'].unique()))
    DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), possible_particle=sorted(df_filtered['particle'].unique()), normalization='predicted', title='Versus Predicted')
    print(f'Running the script required: {time.time()-start}s')
    # Compute feature importance
    '''
        print(f"Feature importance:\n")
        feat_import = clf.tree_.compute_feature_importances(normalize=True)
        feat_import.sort()
        for i in range(len(feat_import)):
            print(features[i],round(100*feat_import[i],2))
        print('-----------------------------------------')
        print(f"Permutation importance:\n")
        perm_import = clf.tree_.compute_feature_importances(normalize=True)
        perm_import.sort()
        for i in range(len(perm_import)):
            print(features[i],round(100*perm_import[i],2))
        print('-----------------------------------------')
    '''