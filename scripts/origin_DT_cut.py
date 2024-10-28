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

np.random.seed(42)

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
    depths = get_depths(inner_tree)
    for index in range(len(inner_tree.value)):
        depth = depths.get(index, 0)
        threshold = np.max([min_threshold, threshold_per_depth * depth])
        if inner_tree.value[index, 0, :8].max() < np.max([threshold, inner_tree.value[index, 0, 8]]) or (inner_tree.weighted_n_node_samples[index] / inner_tree.weighted_n_node_samples[0] < min_samples):
            inner_tree.value[index] *= 0
            inner_tree.value[index, 0, 8] = 1
            
            
def apply_node_threshold(mdl, threshold=0.5, n_tagger=6, min_samples=0):
    inner_tree = mdl.tree_
    for index in range(len(inner_tree.value)):
        if inner_tree.value[index, 0, :8].max() < np.max([threshold, inner_tree.value[index, 0, 8]]) or (inner_tree.weighted_n_node_samples[index] / inner_tree.weighted_n_node_samples[0] < min_samples):
            inner_tree.value[index] *= 0
            inner_tree.value[index, 0, 8] = 1

def reset_nodes(mdl, old_tree):
    inner_tree = mdl.tree_
    for index in range(len(inner_tree.value)):
        if not inner_tree.feature[index] == TREE_UNDEFINED:
            inner_tree.value[index] = old_tree.value[index]
            
def prune_small_leaves(mdl, index=0, min_samples=0.01, bkg_class=-1):
    inner_tree = mdl.tree_
    # if (inner_tree.weighted_n_node_samples[index] / inner_tree.weighted_n_node_samples[0] < min_samples):
    if (inner_tree.n_node_samples[index] / inner_tree.n_node_samples[0] < min_samples):
        inner_tree.value[index, 0, :] *= 0
        inner_tree.value[index, 0, bkg_class] = 1
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        inner_tree.feature[index] = TREE_UNDEFINED
    if not is_leaf(inner_tree, index):
        prune_small_leaves(mdl, inner_tree.children_left[index])
        prune_small_leaves(mdl, inner_tree.children_right[index])
    if index == 0:
        prune_duplicate_leaves(mdl)

    
def check_ambigious_leaves(mdl, index=0, threshold=0.5, bkg_class=-1):
    inner_tree = mdl.tree_
    if (inner_tree.value[index, 0, :].max() < threshold):
        inner_tree.value[index, 0, :] *= 0
        inner_tree.value[index, 0, bkg_class] = 1
    if not is_leaf(inner_tree, index):
        prune_small_leaves(mdl, inner_tree.children_left[index])
        prune_small_leaves(mdl, inner_tree.children_right[index])
    if index == 0:
        prune_duplicate_leaves(mdl)

def summarise_classes(mdl, classes=[], balanced=False, at=None):
    if len(classes) > 1:
        if not at:
            at = classes[0]
        inner_tree = mdl.tree_
        for index in range(len(inner_tree.value)):
            # inner_tree.value[index, 0] = np.append(inner_tree.value[index, 0], 0)
            inner_tree.value[index, 0, at] = np.sum(inner_tree.value[index, 0, classes]) / (1 if not balanced else len(classes))
            inner_tree.value[index, 0, [c for c in classes if c != at]] = 0
        prune_duplicate_leaves(mdl)
    

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
    features = [
        'B_Tr_T_BVIPSig',
        'B_Tr_T_P',
        'B_Tr_T_TRACKISLONG',
        'B_Tr_T_minPhiDistance',
        'B_Tr_T_ISMUON',
        'B_Tr_T_absIP',
        'B_Tr_T_eoverP',
        'B_Tr_T_BPVIPCHI2',
        'B_Tr_T_PT',
        'B_Tr_T_DeltaQ_Pion',
        # 'B_Tr_T_DeltaQ_Proton',
        # 'B_Tr_T_DeltaQ_Kaon',
        'B_Tr_T_Signal_TagPart_PT',
        'B_Tr_T_EtaDistance',
        'B_Tr_T_PhiDistance',
        'B_Tr_T_DeltaR',
        # 'B_Tr_T_Charge',
        'B_Tr_T_cos_PhiDistance',
        'P_proj',
        'B_Tr_T_diff_z',
        # 'B_Tr_T_PX',
        # 'B_Tr_T_PY',
        # 'B_Tr_T_PZ',
        'B_Tr_T_ENERGY',
        'B_Tr_T_Eta',
        'B_Tr_T_Phi',
        'B_Tr_T_BPVIP',
        # 'B_PT',
        # 'B_nTracks',
        'B_Tr_T_MINIP',
        'B_Tr_T_MINIPChi2',
        # 'B_nPVs',
        'B_Tr_T_atanPT_PZ'
    ]
    
    features_pid = [
        'B_Tr_T_PROBNN_PI',
        'B_Tr_T_PROBNN_K',
        'B_Tr_T_PROBNN_E',
        'B_Tr_T_PROBNN_MU',
        'B_Tr_T_PROBNN_P',
        'B_Tr_T_PIDK',
        'B_Tr_T_PIDe',
        'B_Tr_T_PIDmu',
        'B_Tr_T_PIDP',
    ]


    loading_variables = features + features_pid + [
        "B_Tr_T_absID",
        "B_Tr_T_Origin_Flag",
        "B_TRUEID",
        "B_Tr_T_MC_MOTHER_ID",
        "B_Tr_T_MC_GD_MOTHER_ID",
        "B_Tr_T_MC_GD_GD_MOTHER_ID",
        "B_Tr_T_MC_MOTHER_KEY",
        "B_Tr_T_MC_GD_MOTHER_KEY",
        "B_Tr_T_MC_GD_GD_MOTHER_KEY",
        "B_MC_MOTHER_ID",
        "B_MC_GD_MOTHER_ID",
        "B_MC_GD_GD_MOTHER_ID",
        "B_MC_MOTHER_KEY",
        "B_MC_GD_MOTHER_KEY",
        "B_MC_GD_GD_MOTHER_KEY",
        "EVENTNUMBER",
        "RUNNUMBER",
        "B_nPVs",
        "B_BKGCAT",
    ]

    # Path to input root files
    input_paths = {}
    # file_pattern = f'/ceph/users/molocco/classical-taggers/Data/{config.sample_type}/2_added_features/*/*.root'
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/*/*.root'
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bs2DsPi/*1_1.mc.root'#.root'
    # input_paths.update({"Bs2DsPi":glob.glob(file_pattern)})
    file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bu2JpsiK/0023756*2_1.mc.root'#.root'
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bu2JpsiK/0023756*_1.mc.root'#.root'
    input_paths.update({"Bu2JpsiK":glob.glob(file_pattern)})
    file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bd2JpsiKst/0023*2_1.mc.root'#.root'
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bd2JpsiKst/0023*_1.mc.root'#.root'
    input_paths.update({"Bd2JpsiKst":glob.glob(file_pattern)})
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bs2JpsiPhi/*1_1.mc.root'#.root'
    # input_paths.update({"Bs2JpsiPhi":glob.glob(file_pattern)})
    # file_pattern = '/ceph-kernel/users/qfuehring/ft_training_run3/withUT_MC_2024/2_added_features/Bd2DmPi/*1_1.mc.root'#.root'
    # input_paths.update({"Bd2DmPi":glob.glob(file_pattern)})
    #input_paths=['/ceph/users/molocco//classical-taggers/Data/withUT_MC_2024/2_added_features/Bu2JpsiK/00214053_00000002_1.mc.root',]

    print(f"Loading data: Start \n")
    df = pd.DataFrame(columns=loading_variables)
    n = 0
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
                _df["sample"] = n
                n += 1
                # if mode.startswith("Bu"):
                #     _df = _df.query("B_TRUEID==521")
                # elif mode.startswith("Bs"):
                #     _df = _df.query("B_TRUEID==531")
                # elif mode.startswith("Bd"):
                #     _df = _df.query("B_TRUEID==511")
                # _df = _df.query("B_BKGCAT==0")
                # _df = _df.query("B_Tr_T_PT > 500")
                # _df = _df.query("B_nPVs == 1")
            df = pd.concat([df, _df], ignore_index = True)
            print(df.shape, _df.shape)

    print(f"Loading data finished in {round(-start+ time.time() , 2)}s")
    
    
    
    # df = df.query("B_BKGCAT==0 & B_Tr_T_Origin_Flag==1 & B_Tr_T_absID==211").reset_index()
    # for n in range(10):
    #     first_run = df.loc[0, "RUNNUMBER"]
    #     first_event = df["EVENTNUMBER"].unique()[n]
    #     print(first_event, first_run)
    #     _df = df.query(f"sample==0 & RUNNUMBER == {first_run} & EVENTNUMBER == {first_event}").reset_index()
    #     print([(_df.loc[0, f"B_MC{anc}_MOTHER_ID"], _df.loc[0, f"B_MC{anc}_MOTHER_KEY"]) for anc in ["", "_GD", "_GD_GD"]])
    #     for i in range(len(_df)):
    #         print(i, [(_df.loc[i, f"B_Tr_T_MC{anc}_MOTHER_ID"], _df.loc[i, f"B_Tr_T_MC{anc}_MOTHER_KEY"]) for anc in ["", "_GD", "_GD_GD"]])
    # exit(0)
    
    print(df.shape)
    # print(df.groupby(["EVENTNUMBER", "RUNNUMBER", "sample"]).first().shape)
    print(df.query("B_BKGCAT==0").shape[0] / df.shape[0])
    # print(df.groupby(["EVENTNUMBER", "RUNNUMBER", "sample"]).first().shape[0] / df.shape[0])
    df = df.query("B_BKGCAT==0")#.groupby(["EVENTNUMBER", "RUNNUMBER", "sample"]).first()
    print(df.shape)
    
    print(len(df))
    # Shuffle 
    df = df.sample(frac=1, random_state=42)
    df.dropna(inplace=True)

    df.B_Tr_T_Origin_Flag.astype(int)

    # Define labels for multiclassification
    conditions = [
    (df.B_Tr_T_Origin_Flag==2),
    (df.B_Tr_T_Origin_Flag==1) & (df.B_Tr_T_MC_MOTHER_ID.abs() == 5),
    (df.B_Tr_T_Origin_Flag==100), # wrong PV
    # (df.B_Tr_T_Origin_Flag==5), # Prompt 
    ]
    particle_type = { t:i for i, t in enumerate([
        "OS",
        "SS",
        "wrongPV",
        # "Prompt",
        ])
    }
    
    df['ID_type'] = np.select(conditions, particle_type.values(), len(particle_type))
    # ids = df['ID_type'].unique()
    # particle_type = {k:v for k, v in particle_type.items() if v in ids}
    
    particle_type.update({"Other":len(particle_type)})
    # Assign the corresponding particle type
    df['particle'] = df['ID_type'].map({v: k for k, v in particle_type.items()})
    # If ID_type is not in particle_type values, set 'particle' to None
    df.loc[~df['ID_type'].isin(particle_type.values()), 'particle'] = 'unknown'
    
    
    start = time.time()
    print("Start fitting")
    clf = tree.DecisionTreeClassifier(criterion="log_loss", max_depth = 3, min_samples_split=0.1, class_weight='balanced')
    x_train = df[features]
    y_train = df.ID_type
    clf.fit(x_train, y_train)
    
    setting = "category"
    dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True, special_characters=True, proportion=True, node_ids=True, impurity=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(f"{cfg.target_path}/{setting}/tree_schema")
    
    
    prune_duplicate_leaves(clf)
    prune_small_leaves(clf, min_samples=0.05)
    check_ambigious_leaves(clf, threshold=0.5, bkg_class=-1) # tdod different thresholds depending on depth or class
    prune_duplicate_leaves(clf)
    print(f"Fit in: {round(-start+ time.time() , 2)}s\n")
    
    setting = "category"
    dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True, special_characters=True, proportion=True, node_ids=True, impurity=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(f"{cfg.target_path}/{setting}/tree_schema_pruned")
    DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, title='Versus True (pruned)', savepath=f"{cfg.target_path}/{setting}/pruned_confusion_normalised_by_truth.txt")
    DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, normalization='predicted', title='Versus Predicted (pruned)', savepath=f"{cfg.target_path}/{setting}/pruned_confusion_normalised_by_prediction.txt")
    # DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, balanced=True, title='Versus True (balanced)')
    DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, normalization='predicted', title='Versus Predicted (pruned / balanced)', balanced=True, savepath=f"{cfg.target_path}/{setting}/pruned_balanced_confusion_normalised_by_prediction.txt")
    
    df["OS"] = clf.predict(df[features]) == 0
    df["SS"] = clf.predict(df[features]) == 1
    df["wrongPV"] = clf.predict(df[features]) == 2
    df["BKG"] = clf.predict(df[features]) == 3
    
    features += ["OS", "SS", "wrongPV", "BKG"] + features_pid
    df, new_features = add_PID_diffs(df)
    features += new_features
        
    
    
    # Define labels for multiclassification
    conditions = [
    (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==2) & (df.OS==1), # OSKaon
    (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==2) & (df.OS==1), # OSProton
    (df.B_Tr_T_absID==13) & (df.B_Tr_T_Origin_Flag==2) & (df.OS==1),# , # OSMuon
    (df.B_Tr_T_absID==11) & (df.B_Tr_T_Origin_Flag==2) & (df.OS==1) & (df.B_Tr_T_MC_MOTHER_ID!=22), # OSElectron
    (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1) & (df.SS==1),# & (df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# # SSPion
    (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1) & (df.SS==1),# & (df.B_TRUEID.abs()==511)  & (df.B_TRUEID * df.B_Tr_T_Charge < 0), # SSproton
    (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1) & (df.SS==1),# & (df.B_TRUEID.abs()==531) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# SSKaon
    # (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1),# & (df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# # SSPion
    # (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1),# & (df.B_TRUEID.abs()==511)  & (df.B_TRUEID * df.B_Tr_T_Charge < 0), # SSproton
    # (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1),# & (df.B_TRUEID.abs()==531) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# SSKaon
    (df.B_Tr_T_Origin_Flag==100), # wrong PV
    (df.B_Tr_T_absID==11) & (df.B_Tr_T_MC_MOTHER_ID==22),# photon conversion electrons
    (df.B_Tr_T_Origin_Flag==5), # Prompt 
    ]
    particle_type = { t:i for i, t in enumerate([
        "OSKaon",
        "OSProton",
        "OSMuon",
        "OSElectron",
        "SSPion",
        "SSProton",
        "SSKaon",
        # "SSProton+SSKaon",
        "wrongPV",
        "photon conversion",
        "Prompt",
        ])
    }
    
    df['ID_type'] = np.select(conditions, particle_type.values(), len(particle_type))
    # ids = df['ID_type'].unique()
    # particle_type = {k:v for k, v in particle_type.items() if v in ids}
    
    particle_type.update({"Other":len(particle_type)})
    # Assign the corresponding particle type
    df['particle'] = df['ID_type'].map({v: k for k, v in particle_type.items()})
    # If ID_type is not in particle_type values, set 'particle' to None
    df.loc[~df['ID_type'].isin(particle_type.values()), 'particle'] = 'unknown'
    
    print(particle_type)
    
    
    x_train = df[features]
    y_train = df.ID_type
    print(f'The features used are {len(features)}: {features}')
    print(f"\nComposition:\n{round(df.particle.value_counts()/df.shape[0],4)*100}")
    print('-----------------------------------------')
    print(len(y_train))
    for setting in ["balanced"]:#, "unbalanced"]:
        print(setting)
        start = time.time()
        print("Start fitting")
        # Modify loss/score in https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object< similar to https://github.com/keras-team/keras/issues/2115 to weight misID
        if setting == "balanced":
            clf = tree.DecisionTreeClassifier(criterion="log_loss", max_depth = 3, class_weight='balanced',) #class_weight='balanced',  min_impurity_decrease=0.009
        else:
            clf = tree.DecisionTreeClassifier(criterion="log_loss", max_depth = 10, min_samples_leaf=0.002)

        clf.fit(x_train, y_train)
        prune_duplicate_leaves(clf)

        print(f"Fit in: {round(-start+ time.time() , 2)}s\n")
        
        # Visualize the decision tree
        dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True, special_characters=True, proportion=True, node_ids=True, impurity=True) 
        graph = graphviz.Source(dot_data) 
        graph.render(f"{cfg.target_path}/{setting}/tree_schema")
        
        
        # Define labels for multiclassification
        conditions = [
        (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==2) , # OSKaon
        (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==2) , # OSProton
        (df.B_Tr_T_absID==13) & (df.B_Tr_T_Origin_Flag==2),# , # OSMuon
        (df.B_Tr_T_absID==11) & (df.B_Tr_T_Origin_Flag==2)  & (df.B_Tr_T_MC_MOTHER_ID!=22), # OSElectron
        (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1) & (df.B_Tr_T_MC_MOTHER_ID.abs() == 5),# & (df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# # SSPion
        (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1) & (df.B_Tr_T_MC_MOTHER_ID.abs() == 5),# & (df.B_TRUEID.abs()==511)  & (df.B_TRUEID * df.B_Tr_T_Charge < 0), # SSproton
        (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1) & (df.B_Tr_T_MC_MOTHER_ID.abs() == 5),# & (df.B_TRUEID.abs()==531) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# SSKaon
        # (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1),# & (df.B_TRUEID.abs()==511) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# # SSPion
        # (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1),# & (df.B_TRUEID.abs()==511)  & (df.B_TRUEID * df.B_Tr_T_Charge < 0), # SSproton
        # (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1),# & (df.B_TRUEID.abs()==531) & (df.B_TRUEID * df.B_Tr_T_Charge > 0),# SSKaon
        (df.B_Tr_T_Origin_Flag==100), # wrong PV
        (df.B_Tr_T_absID==11) & (df.B_Tr_T_MC_MOTHER_ID==22),# photon conversion electrons
        (df.B_Tr_T_Origin_Flag==5), # Prompt 
        ]
        particle_type = { t:i for i, t in enumerate([
            "OSKaon",
            "OSProton",
            "OSMuon",
            "OSElectron",
            "SSPion",
            "SSProton",
            "SSKaon",
            # "SSProton+SSKaon",
            "wrongPV",
            "photon conversion",
            "Prompt",
            ])
        }
        
        df['ID_type'] = np.select(conditions, particle_type.values(), len(particle_type))
        particle_type.update({"Other":len(particle_type)})
        
        x_train = df[features]
        y_train = df.ID_type
        
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, savepath=f"{cfg.target_path}/{setting}/confusion_normalised_by_truth.txt")
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, normalization='predicted', title='Versus Predicted', savepath=f"{cfg.target_path}/{setting}/confusion_normalised_by_prediction.txt")
        # DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, balanced=True, title='Versus True (balanced)')
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, normalization='predicted', title='Versus Predicted (balanced)', balanced=True, savepath=f"{cfg.target_path}/{setting}/balanced_confusion_normalised_by_prediction.txt")
        
        n_tagger = 7
        # prune_small_leaves(clf, min_samples=0.005)
        # check_ambigious_leaves(clf, threshold=0.33, bkg_class=-1) # tdod different thresholds depending on depth or class
        # try implement pruning based on improvement (purity vs size)
        # try implement pruning threshold based on class confusion
        
        summarise_classes(clf, [5, 6], at=5, balanced=False) # summarise ssk/p classes #todo:fix balancing
        y_train[y_train==6] = 5
        summarise_classes(clf, [8, 9, 10], at=6, balanced=False) # bkg #todo:fix balancing
        y_train[y_train==8] = 6
        y_train[y_train==9] = 6
        y_train[y_train==10] = 6
        summarise_classes(clf, [6, 7], at=6, balanced=False) # sanity #todo:fix balancing
        y_train[y_train==7] = 6
        
        particle_type = {k:v for k, v in particle_type.items() if v not in [5, 6]}
        particle_type.update({"SSKaon / SSProton":5, "Other":6})
        particle_type = dict(sorted(particle_type.items(), key=lambda x: x[1]))
        print(particle_type)
        
        dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True, special_characters=True, proportion=True, node_ids=True, impurity=True) 
        graph = graphviz.Source(dot_data) 
        graph.render(f"{cfg.target_path}/{setting}/tree_schema_pruned")
        
        particle_type = {k:v for k, v in particle_type.items() if v < 6}
        particle_type.update({"Other":6})
        particle_type = dict(sorted(particle_type.items(), key=lambda x: x[1]))
        print(particle_type)
        print("\n Metrics for particle type composition: true VS predicted\n")
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, title='Versus True (pruned)', savepath=f"{cfg.target_path}/{setting}/pruned_confusion_normalised_by_truth.txt")
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, normalization='predicted', title='Versus Predicted (pruned)', savepath=f"{cfg.target_path}/{setting}/pruned_confusion_normalised_by_prediction.txt")
        # DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, balanced=True, title='Versus True (balanced)')
        DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), particle_type_dict=particle_type, normalization='predicted', title='Versus Predicted (pruned / balanced)', balanced=True, savepath=f"{cfg.target_path}/{setting}/pruned_balanced_confusion_normalised_by_prediction.txt")

# maybe consider 3 stage classifier
# 1. OS SS PV, other
# 2. SSpi, OSe, OSmu, K+P, Bkg
# 3. K+P -> OSK, OSP, SSK+P