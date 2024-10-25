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
import copy
# import configParameters as config
import glob
import argparse
# Local import
from IPython import embed
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
plt.rcParams['text.usetex'] = False # HD cluster has some problems with dvp not found
plt.rcParams.update({'axes.unicode_minus' : False})
import DT_utils
from utils import find_tree_name


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
def add_PID_diffs(df):# consider sum or product of multiple (>2) Probnn
    ids = ["e", "mu", "K", "P"]
    l = []
    for i, t in enumerate(ids[:-1]):
        for j in ids[i+1:]:
            df[f"B_Tr_T_PID_{t}-{j}"] = df[f"B_Tr_T_PID{t}"] - df[f"B_Tr_T_PID{j}"]
            df[f"B_Tr_T_PID_{t}+{j}"] = df[f"B_Tr_T_PID{t}"] + df[f"B_Tr_T_PID{j}"]
            # df[f"B_Tr_T_PID_{t}*{j}"] = df[f"B_Tr_T_PID{t}"] * df[f"B_Tr_T_PID{j}"]
            # df[f"B_Tr_T_PID_{t}/{j}"] = df[f"B_Tr_T_PID{t}"] / df[f"B_Tr_T_PID{j}"]
            l += [f"B_Tr_T_PID_{t}-{j}", f"B_Tr_T_PID_{t}+{j}",]# f"B_Tr_T_PID_{t}*{j}", f"B_Tr_T_PID_{t}/{j}"]
    ids = ["E", "MU", "K", "P", "PI"]
    l = []
    for i, t in enumerate(ids[:-1]):
        for j in ids[i+1:]:
            df[f"B_Tr_T_PROBNN_{t}-{j}"] = df[f"B_Tr_T_PROBNN_{t}"] - df[f"B_Tr_T_PROBNN_{j}"]
            df[f"B_Tr_T_PROBNN_{t}+{j}"] = df[f"B_Tr_T_PROBNN_{t}"] + df[f"B_Tr_T_PROBNN_{j}"]
            df[f"B_Tr_T_PROBNN_{t}*{j}"] = df[f"B_Tr_T_PROBNN_{t}"] * df[f"B_Tr_T_PROBNN_{j}"]
            df[f"B_Tr_T_PROBNN_{t}/{j}"] = df[f"B_Tr_T_PROBNN_{t}"] / df[f"B_Tr_T_PROBNN_{j}"]
            l += [f"B_Tr_T_PROBNN_{t}-{j}", f"B_Tr_T_PROBNN_{t}+{j}", f"B_Tr_T_PROBNN_{t}*{j}", f"B_Tr_T_PROBNN_{t}/{j}"]
    # print(l)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, l

def plot_features_byOrigin(data, features, particle_type, nbins=100):
    # Plot input features 
    plt.figure(figsize=(100,100))
    for i, col in enumerate(data.columns.to_list()[:len(features)]):
        plt.subplot(10, 5, i + 1)
        # Ranges and names must be adapted
        #plt.hist(data[col][data['ID_type']==particle_type['OSKaon']], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=)
        #plt.hist(data[col][data['ID_type']==particle_type['OSMuon']], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=ranges[col])
        #plt.hist(data[col][data['ID_type']==particle_type['OSElectron']], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=ranges[col])
        #plt.hist(data[col][data['ID_type']==particle_type['SSPion']], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=ranges[col])
        #plt.hist(data[col][data['ID_type']==particle_type['SSProton']], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=ranges[col])
        #plt.hist(data[col][data['ID_type']==particle_type['SSKaon']], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=ranges[col])
       

        #plt.hist(data[col][data['ID_type']==6], density = True, bins=nbins, label = f"{particle_type[6]}", histtype='step', color='r', )
        if col in nice_names.keys():
            plt.xlabel(nice_names[col])
            plt.hist(data[col][data['ID_type']==particle_type['OSKaon']], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=ranges[col])
            plt.hist(data[col][data['ID_type']==particle_type['OSMuon']], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=ranges[col])
            plt.hist(data[col][data['ID_type']==particle_type['OSElectron']], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=ranges[col])
            plt.hist(data[col][data['ID_type']==particle_type['SSPion']], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=ranges[col])
            plt.hist(data[col][data['ID_type']==particle_type['SSProton']], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=ranges[col])
            plt.hist(data[col][data['ID_type']==particle_type['SSKaon']], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=ranges[col])
        else:
            plt.xlabel(col)
            plt.hist(data[col][data['ID_type']==particle_type['OSKaon']], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, )
            plt.hist(data[col][data['ID_type']==particle_type['OSMuon']], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, )
            plt.hist(data[col][data['ID_type']==particle_type['OSElectron']], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, )
            plt.hist(data[col][data['ID_type']==particle_type['SSPion']], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, )
            plt.hist(data[col][data['ID_type']==particle_type['SSProton']], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, )
            plt.hist(data[col][data['ID_type']==particle_type['SSKaon']], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, )
            plt.hist(data[col][data['ID_type']==particle_type['OSProton']], density = True, bins=nbins, label = f"OSProton", histtype='step', color='orange', lw=2, )
        plt.legend() 
        plt.tight_layout()
    plt.savefig(f"{cfg.target_path}/newPres_DT_features_byOrigin.pdf")

def plot_features_byParticle(data, features, nbins=100):
   
    plt.figure(figsize=(100,100))
    for i, col in enumerate(data.columns.to_list()[:len(features)]):
        plt.subplot(10, 5, i + 1)
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
        #plt.hist(data[col][data['ID_type']==6], density = True, bins=nbins, label = f"{particle_type[6]}", histtype='step', color='r', )
        if col in nice_names.keys():
            plt.xlabel(nice_names[col])
        else:
            plt.xlabel(col)
        plt.legend() 
        plt.tight_layout()
    plt.savefig(f"{cfg.target_path}/newPres_DT_features_byParticle.pdf")

# from https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED

def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and 
            inner_tree.children_right[index] == TREE_LEAF)

def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:     
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[inner_tree.children_left[index]] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        inner_tree.feature[index] = TREE_UNDEFINED
        ##print("Pruned {}".format(index))

def prune_duplicate_leaves(mdl):
    # Remove leaves if both 
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    prune_index(mdl.tree_, decisions)
    
def prune_unclassified_leaves(mdl, unclassfied_entry=6):
    inner_tree = mdl.tree_
    prune_unclassified(inner_tree, unclassfied_entry=unclassfied_entry)
    
def prune_unclassified(inner_tree, index=0, unclassfied_entry=6):
    if (is_leaf(inner_tree, inner_tree[index]) and
        (inner_tree.value[index, 0].argmax() == inner_tree.value[index, unclassfied_entry])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        inner_tree.feature[index] = TREE_UNDEFINED
        ##print("Pruned {}".format(index))
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_unclassified(inner_tree, inner_tree.children_left[index], unclassfied_entry=unclassfied_entry)
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_unclassified(inner_tree, inner_tree.children_right[index], unclassfied_entry=unclassfied_entry)

def get_depths(inner_tree):
    depths = {}
    indices = [0]
    depth = 0
    while len(indices) > 0:
        depths.update({index:depth for index in indices})
        new_indices = []
        for index in indices:
            if index >= 0:
                new_indices += [inner_tree.children_left[index], inner_tree.children_right[index]]
        indices = new_indices
        depth += 1
    # print(depths)
    return depths

def apply_increasing_node_threshold(mdl, min_threshold=0.5, threshold_per_depth=0.1, n_tagger=6, min_samples=0):
    inner_tree = mdl.tree_
    summarise_bkg_classes(mdl, n_tagger)
    depths = get_depths(inner_tree)
    for index in range(len(inner_tree.value)):
        depth = depths.get(index, 0)
        threshold = np.max([min_threshold, threshold_per_depth * depth])
        if inner_tree.value[index, 0, :n_tagger].max() < np.max([threshold, inner_tree.value[index, 0, n_tagger]]) or (inner_tree.weighted_n_node_samples[index] / inner_tree.weighted_n_node_samples[0] < min_samples):
            # inner_tree.value[index] *= 0
            # inner_tree.value[index, 0, n_tagger] = 1
            inner_tree.value[index, 0, n_tagger] = 4 / (3*n_tagger + 4)
            inner_tree.value[index, 0, :n_tagger] = 3 / (3*n_tagger + 4)
            # inner_tree.class = "Unclassified"
            
            
def apply_node_threshold(mdl, threshold=0.5, n_tagger=6, min_samples=0):
    inner_tree = mdl.tree_
    summarise_bkg_classes(mdl, n_tagger)
    for index in range(len(inner_tree.value)):
        if inner_tree.value[index, 0, :n_tagger].max() < np.max([threshold, inner_tree.value[index, 0, n_tagger]]) or (inner_tree.weighted_n_node_samples[index] / inner_tree.weighted_n_node_samples[0] < min_samples):
            # inner_tree.value[index] *= 0
            # inner_tree.value[index, 0, n_tagger] = 1
            inner_tree.value[index, 0, n_tagger] = 4 / (3*n_tagger + 4)
            inner_tree.value[index, 0, :n_tagger] = 3 / (3*n_tagger + 4)
            # inner_tree.class = "Unclassified"

def reset_nodes(mdl, old_tree):
    inner_tree = mdl.tree_
    for index in range(len(inner_tree.value)):
        if not inner_tree.feature[index] == TREE_UNDEFINED:
            inner_tree.value[index] = old_tree.value[index]

def summarise_bkg_classes(mdl, n_tagger=6):
    inner_tree = mdl.tree_
    for index in range(len(inner_tree.value)):
        # inner_tree.value[index, 0] = np.append(inner_tree.value[index, 0], 0)
        inner_tree.value[index, 0, n_tagger] = np.sum(inner_tree.value[index, 0, n_tagger:])
        inner_tree.value[index, 0, (n_tagger+1):] = 0
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='./build/')
    # parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')

    cfg = parser.parse_args()

    from pprint import pprint
    pprint(cfg)
    start = time.time()

    run_time = time.time()
    # Check and eventually make output directory where training info will be saved
    os.makedirs(cfg.target_path, exist_ok=True)

    # set1 --> only uses PIDs and IP significance of the B primary vertex
    # set2 --> collects the pre-selections features used in run2 (except SSKaon). See https://gitlab.cern.ch/lhcb/Phys/-/tree/run2-patches/Phys/FlavourTagging/python/FlavourTagging
    features_set1 = ['B_Tr_T_PROBNN_PI', 'B_Tr_T_PROBNN_K', 'B_Tr_T_PROBNN_E', 'B_Tr_T_PROBNN_MU', 'B_Tr_T_PROBNN_P', 'B_Tr_T_PIDK', 'B_Tr_T_PIDe', 'B_Tr_T_PIDmu', 'B_Tr_T_PIDP', 'B_Tr_T_BVIPSig'] 
    features_set2 = ['B_Tr_T_P' , 'B_Tr_T_TRACKISLONG', 'B_Tr_T_minPhiDistance', 'B_Tr_T_ISMUON', 'B_Tr_T_absIP',
                    'B_Tr_T_eoverP', 'B_Tr_T_BPVIPCHI2', 'B_Tr_T_PT', 'B_Tr_T_DeltaQ_Pion', 'B_Tr_T_DeltaQ_Proton', 'B_Tr_T_DeltaQ_Kaon', 'B_Tr_T_Signal_TagPart_PT', 'B_Tr_T_EtaDistance', 'B_Tr_T_PhiDistance', 'B_Tr_T_DeltaR', 'B_Tr_T_Charge'] # , 'B_Tr_T_CHI2DOF', 'B_Tr_T_GHOSTPROB']
    # set3 --> on the top of set1 and set2 adds other variables from https://gitlab.cern.ch/lhcb/Phys/-/blob/run2-patches/Phys/FlavourTagging/python/FlavourTagging/DevelopmentTaggerConf.py
    features_set3 = ['B_Tr_T_cos_PhiDistance', 'P_proj', 'B_Tr_T_diff_z', 'B_Tr_T_PX', 'B_Tr_T_PY', 'B_Tr_T_PZ', 'B_Tr_T_ENERGY', 'B_Tr_T_Eta', 'B_Tr_T_Phi', 'B_Tr_T_BPVIP', 'B_PT', 'B_nTracks', 'B_Tr_T_MINIP', 'B_Tr_T_MINIPChi2', 'B_nPVs', 'B_Tr_T_atanPT_PZ'] 
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
    input_paths = {}
    # file_pattern = f'/ceph/users/molocco/classical-taggers/Data/{config.sample_type}/2_added_features/*/*.root'
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/*/*.root'
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bs2DsPi/*1_1.mc.root'#.root'
    # input_paths.update({"Bs2DsPi":glob.glob(file_pattern)})
    file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bu2JpsiK/0023756*2_1.mc.root'#.root'
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bu2JpsiK/0023756*_1.mc.root'#.root'
    input_paths.update({"Bu2JpsiK":glob.glob(file_pattern)})
    # # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bd2JpsiKst/0023*2_1.mc.root'#.root'
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bd2JpsiKst/0023*_1.mc.root'#.root'
    # input_paths.update({"Bd2JpsiKst":glob.glob(file_pattern)})
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bs2JpsiPhi/*1_1.mc.root'#.root'
    # input_paths.update({"Bs2JpsiPhi":glob.glob(file_pattern)})
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bd2DmPi/*1_1.mc.root'#.root'
    # input_paths.update({"Bd2DmPi":glob.glob(file_pattern)})
    #input_paths=['/ceph/users/molocco//classical-taggers/Data/withUT_MC_2024/2_added_features/Bu2JpsiK/00214053_00000002_1.mc.root',]

    print(f"Loading data: Start \n")
    df = pd.DataFrame(columns=loading_variables)
    for mode, files in input_paths.items():
        for f in sorted(files):
            print(f"Reading input file: {f}")
            with uproot.open("{}".format(f)) as _f:
                # print(_f[find_tree_name(mode)].keys())
                if mode.startswith("Bu"):
                    loading_variables_temp = [var.replace("B_", "Bu_") for var in loading_variables]
                elif mode.startswith("Bs"):
                    loading_variables_temp = [var.replace("B_", "Bs_") for var in loading_variables]
                elif mode.startswith("Bd"):
                    loading_variables_temp = [var.replace("B_", "Bd_") for var in loading_variables]
                else:
                    loading_variables_temp = loading_variables
                _df = _f[find_tree_name(mode)].arrays(loading_variables_temp, library="pd")
                _df = _df.rename(columns={a:b for a, b in zip(loading_variables_temp, loading_variables)})
                # _df = _df.query("B_BKGCAT==0")
                _df = _df.query("B_Tr_T_PT > 500")
            df = pd.concat([df, _df], ignore_index = True)
            print(df.shape, _df.shape)

    print(f"Loading data finished in {round(-start+ time.time() , 2)}s")
    
    df, new_features = add_PID_diffs(df)
    features += new_features

    df.B_Tr_T_Origin_Flag.astype(int)

    # Define labels for multiclassification
    conditions = [
    (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==2), # OSKaon
    (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==2), # OSProton
    (df.B_Tr_T_absID==13) & (df.B_Tr_T_Origin_Flag==2),# & (df.B_Tr_T_ISMUON == 1), # OSMuon
    (df.B_Tr_T_absID==11) & (df.B_Tr_T_Origin_Flag==2) & (df.B_Tr_T_MC_MOTHER_ID!=22),# & (df.index%2==0), # OSElectron
    (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1) & (df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# & (df.B_Tr_T_DeltaR < 2.5), # SSPion
    ((df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1) & (df.B_TRUEID.abs()==511)  & (df.B_TRUEID * df.B_Tr_T_Charge < 0)) | ((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1) & (df.B_TRUEID.abs()==531) & (df.B_TRUEID * df.B_Tr_T_Charge > 0)),# & (df.B_Tr_T_DeltaR < 2.5), # SSProton and SSKaon
    # (((df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1) & (((df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge < 0)) | (df.B_TRUEID.abs()==531))) | ((df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1) & (((df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge > 0)) | (df.B_TRUEID.abs()!=531))) | ((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1) & (((df.B_TRUEID.abs()==531) & (df.B_TRUEID * df.B_Tr_T_Charge < 0)) | (df.B_TRUEID.abs()!=531)))) & (df.index%4==0), # wrongSS
    # ((df.B_Tr_T_Origin_Flag==3) | (df.B_Tr_T_Origin_Flag==4)) & (df.index%16==0), # OSFrag reduced by a factor 16
    (df.B_Tr_T_Origin_Flag==100) & (df.index%400==0), # wrong PV reduced by a factor 400
    (df.B_Tr_T_absID==11) & (df.B_Tr_T_MC_MOTHER_ID==22) & (df.index%10==0),# & (df.index%2==0), # photon conversion
    (df.B_Tr_T_Origin_Flag==5) & (df.index%150==0), # Prompt reduced by a factor 150
    # ((df.B_Tr_T_Origin_Flag==5) & (df.index%250==0)) | (((df.B_Tr_T_Origin_Flag==3) | (df.B_Tr_T_Origin_Flag==4)) & (df.index%25==0)) | ((((df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1) & (((df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge < 0)) | (df.B_TRUEID.abs()==531))) | ((df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1) & (((df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge > 0)) | (df.B_TRUEID.abs()!=531))) | ((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1) & (((df.B_TRUEID.abs()==531) & (df.B_TRUEID * df.B_Tr_T_Charge < 0)) | (df.B_TRUEID.abs()!=531)))) & (df.index%8==0)) | ((df.B_Tr_T_absID==11) & (df.B_Tr_T_MC_MOTHER_ID==22)), # Other
    # False,# unclassified
    ]
    # conditions.append(exec("("+"|".join([f'!({c.replace(" ", "").split("&(df.index%)")[0]})' for c in conditions])+") & df.index%400==0")) # "other" all tracks which do not belong to one of the defined classes
    particle_type = { t:i+1 for i, t in enumerate([
        "OSKaon",
        "OSProton",
        "OSMuon",
        "OSElectron",
        "SSPion",
        "SSProton+SSKaon",
        # "wrongSS",
        # "OSFragemntation",
        "wrongPV",
        "photon conversion",
        "Prompt",
        # "Other",
        # "Unclassified",
        ])
    }
    
    print(particle_type)
    df['ID_type'] = np.select(conditions, particle_type.values())
    
    # ids = df['ID_type'].unique()
    # particle_type = {k:v for k, v in particle_type.items() if v in ids}
    
    df.loc[~df['ID_type'].isin(particle_type.values()), 'ID_type'] = 0
    # Assign the corresponding particle type
    df['particle'] = df['ID_type'].map({v: k for k, v in particle_type.items()})
    # If ID_type is not in particle_type values, set 'particle' to None
    df.loc[~df['ID_type'].isin(particle_type.values()), 'particle'] = 'not_taggingPart'
    
    # Plot features
    output_dir = 'DT_outputs'
    #plot_features_byOrigin(data=df, features=features, particle_type=particle_type,)
    #plot_features_byParticle(data=df, features=features)
    
    # Shuffle 
    print(len(df))
    df = df.sample(frac=1)
    df.dropna(inplace=True)
    x = df.loc[(df.ID_type != 0 )][features + ["ID_type", "particle"]]
    print(f'The features used are {len(features)}: {features}')
    print(f"\nComposition:\n{round(x.particle.value_counts()/x.shape[0],4)*100}")
    print('-----------------------------------------')
    # To get same amount of not_taggingPart
    #x = pd.concat([x, df.loc[df.ID_type == 0][features + ["ID_type"]].head(len(x))])
    y = x.ID_type
    x.drop(columns=["ID_type", "particle"] , inplace = True)
    print(len(x), len(y))
    x_train, y_train = x, y
    # x_train , x_test ,y_train, y_test= train_test_split(x, y, test_size = 0.3, random_state=42)
    for setting in ["balanced", "unbalanced"]:
        print(setting)
        start = time.time()
        print("Start fitting")
        # Modify loss/score in https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object< similar to https://github.com/keras-team/keras/issues/2115 to weight misID
        # clf = tree.DecisionTreeClassifier(criterion="log_loss", max_depth = 8, min_weight_fraction_leaf=0.01, class_weight='balanced') #class_weight='balanced',  min_impurity_decrease=0.009
        # clf = tree.DecisionTreeClassifier(criterion="log_loss", max_depth = 12, min_samples_leaf=0.01, class_weight='balanced') #class_weight='balanced',  min_impurity_decrease=0.009
        if setting == "balanced":
            clf = tree.DecisionTreeClassifier(criterion="log_loss", max_depth = 10, class_weight='balanced') #class_weight='balanced',  min_impurity_decrease=0.009
        else:
            clf = tree.DecisionTreeClassifier(criterion="log_loss", max_depth = 10, min_samples_leaf=0.002)

        clf.fit(x_train, y_train)
        prune_duplicate_leaves(clf)

        print(f"Fit in: {round(-start+ time.time() , 2)}s\n")
        
        # Visualize the decision tree
        dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True, special_characters=True, proportion=True) 
        graph = graphviz.Source(dot_data) 
        graph.render(f"{cfg.target_path}/{setting}/tree_schema")
        
        n_tagger = 6
        old_tree = copy.deepcopy(clf.tree_)
        old_tree.value[0] *= 0
        while np.any(old_tree.value != clf.tree_.value):
            old_tree = copy.deepcopy(clf.tree_)
            apply_node_threshold(clf, threshold=0.60, n_tagger=n_tagger, min_samples=0 if setting == "unbalanced" else 0.01) # try to implement sample size dependent thresholds
            # apply_increasing_node_threshold(clf, min_threshold=0.50, threshold_per_depth=0.13, n_tagger=n_tagger, min_samples=0 if setting == "unbalanced" else 0.01) 
            prune_duplicate_leaves(clf)
            reset_nodes(clf, old_tree)
            prune_duplicate_leaves(clf)
        
        dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True, special_characters=True, proportion=True) 
        graph = graphviz.Source(dot_data) 
        graph.render(f"{cfg.target_path}/{setting}/tree_schema_pruned")
        
        summarise_bkg_classes(clf, n_tagger)
        particle_type = {k:v for k, v in particle_type.items() if v <= n_tagger}
        particle_type.update({"Rejected":n_tagger+1})
        print(particle_type)
        
        print("\n Metrics for particle type composition: true VS predicted\n")
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, savepath=f"{cfg.target_path}/{setting}/confusion_normalised_by_truth.txt")
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, normalization='predicted', title='Versus Predicted', savepath=f"{cfg.target_path}/{setting}/confusion_normalised_by_prediction.txt")
        # DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, balanced=True, title='Versus True (balanced)')
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, normalization='predicted', title='Versus Predicted (balanced)', balanced=True, savepath=f"{cfg.target_path}/{setting}/balanced_confusion_normalised_by_prediction.txt")
