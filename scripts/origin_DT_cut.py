import glob
import json

import numpy as np 
import pandas as pd 
# from sklearn import tree
import sys 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import graphviz 
from sklearn.metrics import accuracy_score, roc_curve ,auc
import time
import uproot
import os
import argparse
import datetime
from collections import OrderedDict, defaultdict
from sklearn.tree import export_text
import re
import warnings
import yaml

# Local import
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
plt.rcParams['text.usetex'] = False # HD cluster has some problems with dvp not found
plt.rcParams.update({'axes.unicode_minus' : False})
import DT_utils
import shutil
import pyDecisionTree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from pickle import dump, load
from IPython import embed
import seaborn as sns

'''
Ref: https://gitlab.cern.ch/lhcb/Rec/-/blob/master/Phys/DaVinciMCKernel/src/Lib/MCTaggingHelper.cpp?ref_type=heads
Origin Flag IDs:

0 == Signal
1 == SS Fragmentation
2 == OS Decay (track has B0, B+, Bs, Bc+ mother)
3 == OS Fragmentation from excited B
4 == OS Frag from b quark
5 == Prompt
100 == Tracks from other vertex
-1 == No associated MC particle (probably ghost)

'''


def get_balancing_weights(y_train):
    # Count the number of samples in each class
    class_counts = y_train.value_counts()
    total_samples = len(y_train)

    # Calculate weights for each class
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}

    # Map the weights to the original labels in y_train
    weights = y_train.map(class_weights).to_numpy()

    return weights




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Try a Decision Tree for selecting different tagging particle types',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--input_files', help='Pattern for input files', nargs='+', type=str)
    parser.add_argument('--treename', help='Name of the tree in the root files', type=str, default='DecayTree;1')
    parser.add_argument('--target_path', help='Name of the output dir', type=str)
    parser.add_argument('--balanced', help='If classes are balanced or unbalanced', choices=('balanced', 'unbalanced'), type=str, default='balanced')
    parser.add_argument('--unify_SS', help='Whether unify SS taggers into one class at first, then train a secondary selection', action='store_true' )
    parser.add_argument('--SS_unify_weights', help='Only relevant if unify_SS is specified. If specified, the SS class is weighted as a single class.', action='store_true')
    parser.add_argument('--BKG0', help='If specified, only BGKCAT=0 tracks are used',  action='store_true') # Per default BKG0==0 are removed
    parser.add_argument('--load', help='If specified, DT is loaded, instead of trained',  action='store_true')
    parser.add_argument('--all_plots', help='If specified, a histogramm of all variables is plotted',  action='store_true') 
    parser.add_argument('--downsample', help='If specified, the notSamePV class is downsampled to have the same number of tracks as the largest other class (excluding notSamePV)',  action='store_true')
    parser.add_argument('--train_classes', help='Particle types included in the training of the DT, other particle classes are included in the test set for monitoring.', default=["OSKaon", "OSMuon", "OSElectron", "SSPion", "SSProton", "SSKaon", 'notSamePV'], nargs='+', type=str)
    parser.add_argument('--save_dataframes', help='If specified, dataframes containing training and exporatory information is saved to disk for debugging or prototyping',  action='store_true')
    parser.add_argument('--conf_weight_config', help='A dictionary containing the parameter constructing the confusion matrix weights.', type=str,)
    parser.add_argument('--feature_set', help='The set of features to use for training the DT.', default='v1_set', type=str)
    parser.add_argument('--decay_specific_TaggDefinition', help='If specified, the definition of the SS Tagging Particles is decay specific, i.e. SSKaon for Bs and SSPion/Proton for Bd.', action='store_true')
    parser.add_argument('--min_impurity_decrease', help='The minimum impurity decrease required for a split in the DT.', type=float, default=0.009)


    print(f'Run at time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    cfg = parser.parse_args()

    if cfg.unify_SS and cfg.SS_unify_weights:
        warnings.warn(f"SS_unify_weights is specified as {cfg.SS_unify_weights}, but unify_SS is not specified. The SS_unify_weights will be ignored.", UserWarning)



    conf_weights = None
    if cfg.conf_weight_config:
        #Construct the confusion weight matrix if a config is provided

        conf_weight_config = yaml.safe_load(cfg.conf_weight_config)

        conf_weights = np.ones((len(cfg.train_classes),len(cfg.train_classes)), dtype=np.float32)

        if 'diag' in conf_weight_config:
            diag = conf_weight_config.pop('diag')
            for i in range(len(cfg.train_classes)):
                conf_weights[i,i] = diag
        if 'SSKP_as_Rest' in conf_weight_config:
            opposite_dec = conf_weight_config.pop('SSKP_as_Rest')
            opposite_decision_taggers = ['SSKaon', 'SSPion']
            opposite_decision_indices = [cfg.train_classes.index(tagger) for tagger in opposite_decision_taggers if tagger in cfg.train_classes]
            for op_tagg_idx in opposite_decision_indices:    
                for i in range(len(cfg.train_classes)):
                    if i not in opposite_decision_indices:
                        conf_weights[op_tagg_idx,i] = opposite_dec
                        # conf_weights[i,op_tagg_idx] = opposite_dec
        if 'Rest_as_SSKP' in conf_weight_config:
            opposite_dec = conf_weight_config.pop('Rest_as_SSKP')
            opposite_decision_taggers = ['SSKaon', 'SSPion']
            opposite_decision_indices = [cfg.train_classes.index(tagger) for tagger in opposite_decision_taggers if tagger in cfg.train_classes]
            for op_tagg_idx in opposite_decision_indices:    
                for i in range(len(cfg.train_classes)):
                    if i not in opposite_decision_indices:
                        # conf_weights[op_tagg_idx,i] = opposite_dec
                        conf_weights[i,op_tagg_idx] = opposite_dec
        
        if 'tagpart_as_notSamePV' in conf_weight_config:
            tagpart_as_notSamePV = conf_weight_config.pop('tagpart_as_notSamePV')
            tagpart_classes = ['OSKaon', 'OSMuon', 'OSElectron', 'SSKaon', 'SSPion', 'SSProton']
            notSamePV_index = cfg.train_classes.index('notSamePV')
            for tagpart in tagpart_classes:
                if tagpart in cfg.train_classes:
                    tagpart_index = cfg.train_classes.index(tagpart)
                    conf_weights[tagpart_index, notSamePV_index] = tagpart_as_notSamePV
        if 'notSamePV_as_tagpart' in conf_weight_config:
            notSamePV_as_tagpart = conf_weight_config.pop('notSamePV_as_tagpart')
            tagpart_classes = ['OSKaon', 'OSMuon', 'OSElectron', 'SSKaon', 'SSPion', 'SSProton']
            notSamePV_index = cfg.train_classes.index('notSamePV')
            for tagpart in tagpart_classes:
                if tagpart in cfg.train_classes:
                    tagpart_index = cfg.train_classes.index(tagpart)
                    conf_weights[notSamePV_index, tagpart_index] = notSamePV_as_tagpart


        print("Confusion weights:")
        print(conf_weights)
        
        #If any keys are left in the conf_weight_config, they are not recognized and should raise an error
        if conf_weight_config:
            raise NotImplementedError(f"Unrecognized keys in conf_weight_config: {conf_weight_config.keys()}. Only 'diag', 'SSKP_as_Rest', 'Rest_as_SSKP, 'tagpart_as_notSamePV', 'notSamePV_as_tagpart' are currently supported.")


    from pprint import pprint   
    pprint(cfg)
    start = time.time()
    # Check and eventually make output directory where training info will be saved
    os.makedirs(cfg.target_path, exist_ok=True)
    output_path = f'{cfg.target_path}'


    

        
    # Missing fetaures wrt Run2
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


    with open(f'configs/DT_feature_set.yaml', 'r') as f:
        features = yaml.safe_load(f)[cfg.feature_set]


    mc_info = ["B_BKGCAT", "B_Tr_T_absID", "B_Tr_T_Origin_Flag", "B_TRUEID", "B_Tr_T_MC_MOTHER_ID", 
               'B_Tr_T_MC_MOTHER_KEY', 'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_KEY', 
               'B_Tr_T_MC_GD_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_KEY']

    load_extra = ['EVENTNUMBER','RUNNUMBER']
    loading_variables = features + mc_info + load_extra
 
    folders = ['Bd2JpsiKst', 'Bs2DsPi', 'Bu2JpsiK']

    # Iterate over each folder and collect the root files
    print(f"Loading data: Start \n", flush=True)

    df = pd.DataFrame(columns=loading_variables)

    print(df.columns)

    for f in cfg.input_files:
        decay = os.path.basename(os.path.dirname(f))
        print(f"Reading input file: {f}", flush=True)
        with uproot.open("{}".format(f)) as _f:
            _df = _f[cfg.treename].arrays(loading_variables, library="pd")#[:100_000]  # Limit the number of rows for testing TODO REMOVE
            _df['decay'] = decay
            #print(f'Number of tracks per file: {_df.shape[0]}', flush=True)
            df = pd.concat([df, _df], ignore_index = True)
            #print(f'Number of tracks in concatenated df {df.shape[0]}', flush=True)
    df.dropna(inplace=True)
    print(f"Total number of tracks (all decays, all particles): {df.shape[0]}", flush=True)

    df['B_Tr_T_Origin_Flag'] = df['B_Tr_T_Origin_Flag'].astype(int)
    df['B_Tr_T_MC_MOTHER_ID'] = df['B_Tr_T_MC_MOTHER_ID'].astype(int)
    # Define labels for multiclassification
    # List of (condition, particle_type) tuples


    # Last Term  of SS Taggers either always evaluates true if the old definition of tagging particles, i.e. non decay specific, is used 
    # or in the other case it pulls only SSPion and SSProton for Bd2JpsiKst and SSKaon for Bs2DsPi
    # conditions = [
    # ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 2),                                                                           "OSKaon"),
    # ((df.B_Tr_T_absID ==   13) & (df.B_Tr_T_Origin_Flag == 2),                                                                           "OSMuon"),
    # ((df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) !=  22),                                    "OSElectron"),
    # ((df.B_Tr_T_absID ==  211) & (df.B_Tr_T_Origin_Flag == 1),                                                                           "SSPion"), 
    # ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1),                                                                           "SSProton"),
    # ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 1),                                                                           "SSKaon"),
    # ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag       != 100),                                    "otherK"),
    # ((df.B_Tr_T_absID ==   13) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag       != 100),                                    "otherMu"),
    # ((df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag != 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) !=  22) & (df.B_Tr_T_Origin_Flag != 100),   "otherE"),
    # ((df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) ==  22),                                    "photonOSEl"),
    # ((df.B_Tr_T_absID ==  211) & (df.B_Tr_T_Origin_Flag != 1) & (df.B_Tr_T_Origin_Flag       != 100),                                    "otherPi"),
    # ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag != 1) & (df.B_Tr_T_Origin_Flag       != 100),                                    "otherP"),
    
    # ((df.B_Tr_T_Origin_Flag == 100), "notSamePV"),
    # ]

    tagPart_conditions = OrderedDict({
        "OSKaon" : 
            (df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 2),
        "OSMuon" :
            (df.B_Tr_T_absID ==   13) & (df.B_Tr_T_Origin_Flag == 2),
        "OSElectron" :
            (df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) !=  22),
        "SSPion" :
            (df.B_Tr_T_absID ==  211) & (df.B_Tr_T_Origin_Flag == 1),
        "SSProton" :
            (df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1),
        "SSKaon" :
            (df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 1),

        "PhotonOSEl" :
            (df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) ==  22),
        "notSamePV" :
            (df.B_Tr_T_Origin_Flag == 100),
    })
    tagPart_conditions.update({
        "otherK" : 
            (df.B_Tr_T_absID ==  321) & ~tagPart_conditions["notSamePV"] & ~tagPart_conditions["OSKaon"]     & ~tagPart_conditions["SSKaon"],
        "otherE" :
            (df.B_Tr_T_absID ==   11) & ~tagPart_conditions["notSamePV"] & ~tagPart_conditions["OSElectron"] & ~tagPart_conditions["PhotonOSEl"],
        "otherMu" :
            (df.B_Tr_T_absID ==   13) & ~tagPart_conditions["notSamePV"] & ~tagPart_conditions["OSMuon"],
        "otherPi" :
            (df.B_Tr_T_absID ==  211) & ~tagPart_conditions["notSamePV"] & ~tagPart_conditions["SSPion"],
        "otherP" :
            (df.B_Tr_T_absID == 2212) & ~tagPart_conditions["notSamePV"] & ~tagPart_conditions["SSProton"],
    })

    # SSDecSpec_conditions = [
    # ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 2),                                                                         "OSKaon"),
    # ((df.B_Tr_T_absID ==   13) & (df.B_Tr_T_Origin_Flag == 2),                                                                         "OSMuon"),
    # ((df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) !=  22),                                  "OSElectron"),
    # ((df.B_Tr_T_absID ==  211) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay == 'Bd2JpsiKst'),                                            "SSPion"), 
    # ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay == 'Bd2JpsiKst'),                                            "SSProton"),
    # ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay == 'Bs2DsPi'   ),                                            "SSKaon"),
    # # ((df.B_Tr_T_absID ==  321) &((df.B_Tr_T_Origin_Flag != 2) | (df.B_Tr_T_Origin_Flag != 1) | ((df.B_Tr_T_Origin_Flag == 1) & (df.decay != 'Bs2DsPi'   )))          & (df.B_Tr_T_Origin_Flag != 100), "otherK"),
    # ((df.B_Tr_T_absID ==   13) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag       != 100),                                  "otherMu"),
    # ((df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag != 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) !=  22) & (df.B_Tr_T_Origin_Flag != 100), "otherE"),
    # ((df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) ==  22),                                  "photonOSEl"),
    # ((df.B_Tr_T_absID ==  211) &((df.B_Tr_T_Origin_Flag != 1) | (df.decay != 'Bd2JpsiKst'))          & (df.B_Tr_T_Origin_Flag != 100) ,"otherPi"),
    # ((df.B_Tr_T_absID == 2212) &((df.B_Tr_T_Origin_Flag != 1) | (df.decay != 'Bd2JpsiKst'))          & (df.B_Tr_T_Origin_Flag != 100) ,"otherP"),
    
    # ((df.B_Tr_T_Origin_Flag == 100), "notSamePV"),
    # ]

    SSDecSpec_conditions = OrderedDict({
        "OSKaon" : 
            (df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 2),
        "OSMuon" :
            (df.B_Tr_T_absID ==   13) & (df.B_Tr_T_Origin_Flag == 2),
        "OSElectron" :
            (df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) !=  22),
        "SSPion" :
            (df.B_Tr_T_absID ==  211) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay == 'Bd2JpsiKst'),
        "SSProton" :
            (df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay == 'Bd2JpsiKst'),
        "SSKaon" :
            (df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay == 'Bs2DsPi'),

        "PhotonOSEl" :
            (df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) ==  22),
        "notSamePV" :
            (df.B_Tr_T_Origin_Flag == 100),
    })
    SSDecSpec_conditions.update({
        "otherK" : 
            (df.B_Tr_T_absID ==  321) & ~SSDecSpec_conditions["notSamePV"] & ~SSDecSpec_conditions["OSKaon"]     & ~SSDecSpec_conditions["SSKaon"],
        "otherE" :
            (df.B_Tr_T_absID ==   11) & ~SSDecSpec_conditions["notSamePV"] & ~SSDecSpec_conditions["OSElectron"] & ~SSDecSpec_conditions["PhotonOSEl"],
        "otherMu" :
            (df.B_Tr_T_absID ==   13) & ~SSDecSpec_conditions["notSamePV"] & ~SSDecSpec_conditions["OSMuon"],
        "otherPi" :
            (df.B_Tr_T_absID ==  211) & ~SSDecSpec_conditions["notSamePV"] & ~SSDecSpec_conditions["SSPion"],
        "otherP" :
            (df.B_Tr_T_absID == 2212) & ~SSDecSpec_conditions["notSamePV"] & ~SSDecSpec_conditions["SSProton"],
    })


    if cfg.decay_specific_TaggDefinition:
        target_conditions = SSDecSpec_conditions
        target_column = 'decSpec_particle'
    else:
        target_conditions = tagPart_conditions
        target_column = 'particle'
    
    if cfg.unify_SS:
        #combine the SSKaon and SSProton classes into a single class "SSKaon+SSProton"

        SS_conditions = []
        for particle, condition in target_conditions.items():
            if particle.startswith("SS"):
                SS_conditions.append(condition)
    
        # target_conditions = [pair for pair in conditions if not pair[1].startswith("SS")]
        # comb_SS_conditions = (np.logical_or.reduce(SS_conditions), "SS")
        # target_conditions.append(comb_SS_conditions)

        target_conditions = {k: v for k, v in target_conditions.items() if not k.startswith("SS")}
        target_conditions["SS"] = np.logical_or.reduce(SS_conditions)

        cfg.train_classes = [i for i in cfg.train_classes if not i.startswith("SS")] + ["SS"]

        print(f"cfg.train_classes after unifying SS classes: {cfg.train_classes}", flush=True)

    # Separate conditions and particle types for np.select()
    conditions = [condition for _, condition in tagPart_conditions.items()]
    particle_type = [particle for particle, _ in tagPart_conditions.items()]
    # Assign particle types based on conditions, with default "Others" for unmatched rows
    df['particle'] = np.select(conditions, particle_type, default="Others")
    

    conditions = [condition for _, condition in SSDecSpec_conditions.items()]
    particle_type = [particle for particle, _ in SSDecSpec_conditions.items()]
    # Assign particle types based on conditions, with default "Others" for unmatched rows
    df['decSpec_particle'] = np.select(conditions, particle_type, default="Others")

    conditions = [condition for _, condition in target_conditions.items()]
    particle_type = [particle for particle, _ in target_conditions.items()]
    df['target'] = np.select(conditions, particle_type, default="Others")


    print(f"Number of tracks for each particle type:\n{df[target_column].value_counts()}", flush=True)

    print(f"Number of tracks for each target class:\n{df['target'].value_counts()}", flush=True)


    particle_dict = {"OSKaon": 0,
                    "OSMuon": 1,
                    "OSElectron": 2,
                    "SSPion": 3,
                    "SSProton": 4,
                    "SSKaon": 5,
                    "otherK": 6,
                    "otherMu": 7,
                    "otherE": 8,
                    "photonOSEl": 9,
                    "otherPi": 10,
                    "otherP": 11,
                    "notSamePV": 12,
                    "Others": 13,
                    "SS": 14}
    reverse_particle_dict = {v: k for k, v in particle_dict.items()}
    decay_dict = {"Bd2JpsiKst": 0, 
                "Bs2DsPi": 1, 
                "Bu2JpsiK": 2}
    #Save the dataframe with ids, flags and particles types to a root file for exploration
    if cfg.save_dataframes:
        with uproot.recreate(f"{cfg.target_path}/ids_flags_particles.root") as file:
            file["DecayTree"] = df[mc_info + ['particle', 'decSpec_particle', 'target']]
        
        df_export = df.copy()
        # convert all columns, except "particle", to float for better readability in c++
        # For string columns translate to integers first
        print(df_export['particle'].unique())
        df_export['particle'        ] = df_export['particle'        ].map(particle_dict)
        df_export['decSpec_particle'] = df_export['decSpec_particle'].map(particle_dict)
        df_export['target'          ] = df_export['target'          ].map(particle_dict)



        df_export['decay'] = df_export['decay'].map(decay_dict)

        df_export = df_export.astype(float)
        print(df_export['particle'].unique())
        df_export['particle'] = df_export['particle'].astype(int)

        with uproot.recreate(f"{cfg.target_path}/DT10k_trainingset.root") as file:
            file["DecayTree"] = df_export[:10_000]
        with uproot.recreate(f"{cfg.target_path}/DT1M_trainingset.root") as file:
            file["DecayTree"] = df_export[:1_000_000]
        with uproot.recreate(f"{cfg.target_path}/DT_trainingset.root") as file:
            file["DecayTree"] = df_export
        print('Exploration dataframe with particle types, IDs and flags saved to root file', flush=True)
        del df_export


    print(f"Total number of tracks for each B candidate (by TRUE_ID):\n{df['B_TRUEID'].value_counts()}", flush=True)

    # Leaving it as an option, but only BKG_CAT==0 should be the default
    if cfg.BKG0:
        df['B_BKGCAT'] = df['B_BKGCAT'].astype(int)
        print("Filtering tracks based on decay and B_BKGCAT values...", flush=True)

        mask = (
            (df['decay'].str.contains('Bs2DsPi') & (df['B_BKGCAT'] == 20)) # Only for Bs2DsPi due to problems with the BKG_CAT
            | (~df['decay'].str.contains('Bs2DsPi') & (df['B_BKGCAT'] == 0))
        )
        df = df.loc[mask]
        print(f"New number of tracks: {df.shape[0]}", flush=True)
    else:
        DT_utils.count_BKGCAT(df)

    print(f"\nComposition (%) before splitting in training-test set:\n{round(df['target'].value_counts()/df.shape[0],4)*100}", flush=True)
    print(f"\nComposition before splitting in training-test set:\n{df['target'].value_counts()}", flush=True)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    label_vars = ["target", "particle", "decSpec_particle"]

    x = df[features + label_vars]
    del df

    y = x[label_vars].copy()

    print(f'The features used are {len(features)}: {features}', flush=True)
    print('-----------------------------------------', flush=True)
    # To get same amount of not_taggingPart
    x.drop(columns=label_vars , inplace = True)
    X_train , x_test ,Y_train, y_test= train_test_split(x, y, test_size = 0.01, random_state=42)
    del x, y
    print(type(X_train))
    # Drop 'Other' particles from training dataset
    # Drop all particles that are not in train_particle_types, other particles are included for monitoring during testing
    mask = np.isin(Y_train["target"], cfg.train_classes)

    X_train = X_train[mask]
    Y_train = Y_train[mask]

    print(f"Composition before downsampling:\n{Y_train[target_column].value_counts()}", flush=True)

    if cfg.downsample:
        # Downsample all non tagging particle classes (this case only notSamePV, if other background classes are added in the future add them here) 
        # to have the same number of tracks as the largest tagging particle class (which is SSPion)
        # Apply downsampling to the training set only.

        # Downsample the 'notSamePV' classes. 
        # Get the count of the largest class excluding "notSamePV"
        X_train[label_vars] = Y_train[label_vars]

        # max_class_size = X_train[X_train[target_column] == 'SSPion'].value_counts().max()
        max_class_size = X_train.loc[X_train[target_column] != 'notSamePV', target_column].value_counts().max()

        # Filter the 'notSamePV' rows
        not_same_pv_rows = X_train[X_train[target_column] == 'notSamePV']
        # Randomly sample the maximum class size from 'notSamePV'
        sampled_not_same_pv = not_same_pv_rows.sample(n=max_class_size, random_state=42)
        # Filter out 'notSamePV' from the original dataframe to keep the other rows
        X_train = X_train.loc[(X_train[target_column] != 'notSamePV')]
        # Concatenate the sampled 'notSamePV' rows back with the other classes
        X_train = pd.concat([X_train, sampled_not_same_pv])

        Y_train = X_train[label_vars]
        X_train.drop(columns=label_vars, inplace=True)

        #df = pd.concat([df, others_rows])
        print(f'Composition after downsampling:\n{Y_train[target_column].value_counts()}', flush=True)

        del not_same_pv_rows, sampled_not_same_pv

    y_train = Y_train["target"]
    



    print(f"Classes used in training set: {y_train.unique()}", flush=True)

    print(f"Number of tracks in training set: {X_train.shape[0]}", flush=True)
    print(f"Composition of training set:\n{y_train.value_counts()}", flush=True)

    print(f"Number of tracks in test set: {x_test.shape[0]}", flush=True)
    print(f"Composition of test set:\n{y_test.value_counts()}", flush=True)
    
    # if cfg.balanced == 'unbalanced':
    #     weights = None
    # else:
    #     weights = str(cfg.balanced)
 
    if cfg.load:
        print("Loading the model...", flush=True)
        start_load = time.time()
        # with open(f"{output_path}/decision_tree_model.pkl", "rb") as f:
        #     clf = pickle.load(f)

        clf = pyDecisionTree.DecisionTree.load_tree(f"{output_path}/decision_tree_model.yaml")

        # Snakemake expects a txt file for the decision tree which is deleted anytime the snakemake rule is run.
        # The yaml remains allowing to load the model without retraining it.
        shutil.copyfile(f"{output_path}/decision_tree_model.yaml", f"{output_path}/decision_tree_model.txt")

        print(f"Model loaded successfully! Type of clf: {type(clf)}", flush=True)
        print(f'Loading the Decision Tree required: {round(time.time()-start_load, 2)}s', flush=True)

    else:
        # Plot features
        if cfg.all_plots:
            print("Plotting features...", flush=True)
            DT_utils.plot_features_byOrigin(X_train, features, target_path=cfg.target_path, nbins=50)   

        print("Start fitting", flush=True)
        start_fit = time.time()
        # clf = tree.DecisionTreeClassifier(max_depth = 6,class_weight=weights, min_impurity_decrease=0.009)
        # clf.fit(X_train, y_train)
        
        X_train = X_train.astype(np.float32)
        
        y_train_cpp = y_train.to_numpy()
        # Change y_train to in32 for c++ compatibility, since pyDecisionTree expects the target to be of type int32
        # use the particle_dict to convert the particle types to integers
        y_train_cpp = np.array([particle_dict[particle] for particle in y_train_cpp])
        y_train_cpp = y_train_cpp.astype(np.int32)
        particle_labels = [i for i in particle_dict.keys()]

        print(f"particle_labels: {particle_labels}", flush=True)


        #Dont use signal_is_Bs as a feature for a tree which include OS taggers, since they should be signal agnostic.
        features = [f for f in features if f != 'signal_is_Bs']

        cpp_df = pyDecisionTree.Dataframe(X_train[features], y_train_cpp, features, particle_labels, debugInfo=False)

        if cfg.balanced == 'unbalanced':
            balancing_weights = None
        else:

            if cfg.SS_unify_weights:
                balancing_weights = cpp_df.get_balancing_weights()
            else:
                balancing_weights = get_balancing_weights(Y_train[target_column])

        print(f"Balancing weights: {balancing_weights}", flush=True)

        if conf_weights is None:
            criterion = pyDecisionTree.Gini()
        else:
            criterion = pyDecisionTree.Gini_conf_weighted(conf_weights)

        clf = pyDecisionTree.DecisionTree(criterion, 6, X_train.columns.to_list(), particle_labels, 2, 1, 0.0, cfg.min_impurity_decrease, verbose=False)
        clf.fit(cpp_df.get_data(), cpp_df.get_targets(), balancing_weights)
        del cpp_df, y_train_cpp

        clf.save_tree(f"{output_path}/decision_tree_model.yaml")
        # Copy the tree to a txt file as well. Snakemake expects a txt file for the decision tree which is deleted anytime the snakemake rule is run.
        # The yaml remains allowing to load the model without retraining it.
        shutil.copyfile(f"{output_path}/decision_tree_model.yaml", f"{output_path}/decision_tree_model.txt")
        if os.path.exists(f"{output_path}/tree_schema.pdf"):
            os.remove(f"{output_path}/tree_schema.pdf")
        clf.plot_tree(f"{output_path}/tree_schema.pdf")

        if cfg.unify_SS:
            print("Training secondary SS classifier...", flush=True)

            #Training secondary SS classifier to distinguish between SSKaon, SSProton and SSPion
            XSS_train = X_train[y_train == 'SS'].to_numpy()
            yss_train = Y_train[y_train == 'SS'][target_column].to_numpy()

            print(f"yss_train: {yss_train}", flush=True)
            print(f"unqiue yss_train: {np.unique(yss_train)}", flush=True)

            yss_train_cpp = np.array([particle_dict[particle] for particle in yss_train])

            #Getting unique patricle labels in order of their integer representation in the cpp classifier
            # particle_labels_ss = [reverse_particle_dict[i] for i in np.unique(yss_train_cpp)]

            # print(f"particle_labels_ss: {particle_labels_ss}", flush=True)

            cpp_df_ss = pyDecisionTree.Dataframe(XSS_train, yss_train_cpp, X_train.columns.to_list(), particle_labels, debugInfo=False)

            if cfg.balanced == 'unbalanced':
                balancing_weights_ss = None
            else:
                balancing_weights_ss = cpp_df_ss.get_balancing_weights()

            clf_ss = pyDecisionTree.DecisionTree(criterion, 6, X_train.columns.to_list(), particle_labels, 2, 1, 0.0, cfg.min_impurity_decrease, verbose=False)
            clf_ss.fit(cpp_df_ss.get_data(), cpp_df_ss.get_targets(), balancing_weights_ss)

            clf_ss.save_tree(f"{output_path}/decision_tree_model_SS.yaml")
            clf_ss.plot_tree(f"{output_path}/tree_schema_SS.pdf")

            del cpp_df_ss, yss_train_cpp, XSS_train, yss_train


        print(f'Decision Tree training required: {round(time.time()-start_fit, 2)}s', flush=True)
        os.makedirs(output_path, exist_ok=True)


        print("Model saved successfully!", flush=True)
    # Get all decision paths from the classifier

    clf.export_cuts(os.path.join(output_path, "cuts/"))

    if cfg.unify_SS:
        clf_ss.export_cuts(os.path.join(output_path, "cuts_SS/"))
        print(f"Decision paths for secondary SS classifier saved to {output_path}/cuts_SS/", flush=True)


    cut_files = glob.glob(os.path.join(output_path, "cuts/*.txt"))
    print(cut_files)


    paths_by_class = defaultdict(list)
    for cut_file in cut_files:
        with open(cut_file, "r") as f:
            cuts = f.readlines()

        #remove all "OR" lines from cuts
        cuts = [cut.strip() for cut in cuts if cut.strip() != "OR"]

        class_label = os.path.basename(cut_file).replace(".txt", "")
        paths_by_class[class_label] = cuts

    if cfg.unify_SS:
        #Combine the cuts for the secondary SS classifier into the main paths_by_class dictionary
        #The predicted SS class is replaced by the predicted SSKaon, SSProton and SSPion classes
        #Then the cut_files are replaced with the complete list of cuts for each SS particle type
        cut_files_SS = glob.glob(os.path.join(output_path, "cuts_SS/*.txt"))
        paths_by_class_SS = defaultdict(list)
        for cut_file in cut_files_SS:
            with open(cut_file, "r") as f:
                cuts = f.readlines()

            #remove all "OR" lines from cuts
            cuts = [cut.strip() for cut in cuts if cut.strip() != "OR"]

            class_label = os.path.basename(cut_file).replace(".txt", "")
            paths_by_class_SS[class_label] = cuts

        SS_cuts = paths_by_class.pop("SS", [])
        for ss_particle in ["SSKaon", "SSProton", "SSPion"]:
            if ss_particle in paths_by_class_SS:
                #Add SS cuts to the ss_particles cuts
                ss_particle_cuts = []
                for ss_cut in SS_cuts:
                    for ss_sub_cut in paths_by_class_SS[ss_particle]:
                        combined_cut = f"{ss_cut} & {ss_sub_cut}"
                        ss_particle_cuts.append(combined_cut)
                paths_by_class[ss_particle] = ss_particle_cuts


        #remove the previous cut files and replace them with the new combined cuts for each SS particle type
        os.remove(os.path.join(output_path, "cuts/SS.txt"))

        for ss_particle in ["SSKaon", "SSProton", "SSPion"]:
            with open(os.path.join(output_path, f"cuts/{ss_particle}.txt"), "w") as f:
                for i, cut in enumerate(paths_by_class[ss_particle]):
                    f.write(f"{cut}\n")

                    if i < len(paths_by_class[ss_particle]) - 1:
                        f.write("OR\n")






    print(json.dumps(paths_by_class, indent=4), flush=True)

    print(f"Number of paths for each class:\n", flush=True)
    for label, conditions_list in paths_by_class.items():
        print(f"  {label}: {len(conditions_list)}", flush=True)

    #Get all features used by the DT
    features_DT_used = set()
    for label, conditions_list in paths_by_class.items():
        for conditions in conditions_list:
            #removes every character in the conditions that are not the names of variables or sequences of numbers and splits into list of tokens
            tokens = re.sub(r'[<>()!=&.-]', '', conditions).split() 
            # discard tokens that are purely numbers
            conditions_names = [t for t in tokens if not t.isdigit()]

            features_DT_used.update(conditions_names)
    features_DT_used = list(features_DT_used)
    features_DT_used.remove('B_Tr_T_Origin_Flag')
    X_train['particle'] = y_train
    DT_utils.plot_used_features(X_train, features_DT_used, target_path=cfg.target_path, nbins=50)
    print(f"Features used by the Decision Tree:\n {features_DT_used}", flush=True)
    del X_train, y_train




    print("Metrics for particle type composition: true VS predicted\n", flush=True)
    unify_classes = ["otherK", "otherMu", "otherE", "photonOSEl", "otherPi", "otherP"]

    print('Beginning prediction on test set...', flush=True)
    y_pred_test = clf.predict(x_test)

    if cfg.unify_SS:
        # For the predicted SS class, use the secondary classifier to predict the specific SS particle type
        ss_mask = (y_pred_test == 14)
        if np.any(ss_mask):
            x_test_ss = x_test[ss_mask]
            y_pred_ss = clf_ss.predict(x_test_ss)
            y_pred_test[ss_mask] = y_pred_ss

    inv_particle_dict = {v: k for k, v in particle_dict.items()}
    y_pred_test = np.array([inv_particle_dict[pred] for pred in y_pred_test])

    print(f"Prediction on test set completed. Number of predictions: {len(y_pred_test)}", flush=True)
    print(f"Unique predicted classes: {np.unique(y_pred_test)}", flush=True)
    print(f"Predicted classes: {y_pred_test}", flush=True)

    print(f"True classes: {y_test[target_column]}", flush=True)

    #Define order of particles in table/heatmap
    ordered_particles = ['OSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton', 'SSKaon','notSamePV', 'otherK', 'otherMu', 'otherE', 'photonOSEl', 'otherPi', 'otherP', 'Others']
    #Add in any potenially missing particles in the dataset (e.g. if some particle types are not present in the ordererd list but are in the dataset)
    for p in np.unique(y_test[target_column]):
        if p not in ordered_particles:
            ordered_particles.append(p)


    DT_utils.metric_table(y_true=y_test['particle'], y_predicted=y_pred_test, possible_particle=ordered_particles,                            title='Versus True (pruned)',                 savepath=f"{output_path}/allSS_unified_confusion_TruthNormalised.txt",      unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_test['particle'], y_predicted=y_pred_test, possible_particle=ordered_particles, normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/allSS_unified_confusion_PredictionNormalised.txt", unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_test['particle'], y_predicted=y_pred_test, possible_particle=ordered_particles,                            title='Versus True (pruned)',                 savepath=f"{output_path}/allSS_confusion_TruthNormalised.txt",              unify_classes=None)
    DT_utils.metric_table(y_true=y_test['particle'], y_predicted=y_pred_test, possible_particle=ordered_particles, normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/allSS_confusion_PredictionNormalised.txt",         unify_classes=None)


    DT_utils.metric_table(y_true=y_test['decSpec_particle'], y_predicted=y_pred_test, possible_particle=ordered_particles,                            title='Versus True (pruned)',                 savepath=f"{output_path}/SpecSS_unified_confusion_TruthNormalised.txt",      unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_test['decSpec_particle'], y_predicted=y_pred_test, possible_particle=ordered_particles, normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/SpecSS_unified_confusion_PredictionNormalised.txt", unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_test['decSpec_particle'], y_predicted=y_pred_test, possible_particle=ordered_particles,                            title='Versus True (pruned)',                 savepath=f"{output_path}/SpecSS_confusion_TruthNormalised.txt",              unify_classes=None)
    DT_utils.metric_table(y_true=y_test['decSpec_particle'], y_predicted=y_pred_test, possible_particle=ordered_particles, normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/SpecSS_confusion_PredictionNormalised.txt",         unify_classes=None)


    print(f'Running the script required: {time.time()-start}s', flush=True)
