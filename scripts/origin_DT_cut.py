import numpy as np 
import pandas as pd 
from sklearn import tree
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
from collections import defaultdict
from sklearn.tree import export_text
import re

# Local import
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
plt.rcParams['text.usetex'] = False # HD cluster has some problems with dvp not found
plt.rcParams.update({'axes.unicode_minus' : False})
import DT_utils

from IPython import embed
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Try a Decision Tree for selecting different tagging particle types',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--input_files', help='Pattern for input files', nargs='+', type=str)
    parser.add_argument('--treename', help='Name of the tree in the root files', type=str, default='DecayTree;1')
    parser.add_argument('--target_path', help='Name of the output dir', type=str)
    parser.add_argument('--balanced', help='If classes are balanced or unbalanced', choices=('balanced', 'unbalanced'), type=str, default='balanced')
    parser.add_argument('--unify_SS', help='If unify SSKaon and SSProton in a single class', action='store_true' ) # action='store_true' means args.unify_SS will be set to True if the --unify_SS argument is provided on the command line.
    # Per default BKG0==0 are removed
    parser.add_argument('--BKG0', help='If specified, only BGKCAT=0 tracks are used',  action='store_true')
    parser.add_argument('--load', help='If specified, DT is loaded, instead of trained',  action='store_true')
    parser.add_argument('--all_plots', help='If specified, a histogramm of all variables is plotted',  action='store_true') 

    print(f'Run at time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    cfg = parser.parse_args()

    from pprint import pprint   
    pprint(cfg)
    downsampling = False
    start = time.time()
    # Check and eventually make output directory where training info will be saved
    os.makedirs(cfg.target_path, exist_ok=True)
    if cfg.unify_SS:
        output_path = f'{cfg.target_path}/SSKSSP/'
    else:
        output_path = f'{cfg.target_path}'

    features_added = ['B_Tr_T_cos_PhiDistance', 'B_Tr_T_PhiDistance', 'B_Tr_T_diff_z', 'B_Tr_T_DeltaR', 'diff_P', 'P_proj', 't', 'EVIP', 'B_Tr_T_absOWNPV_IP', 'B_Tr_T_EtaDistance', 'B_Tr_T_DeltaQ_Pion', 'B_Tr_T_DeltaQ_Muon', 'B_Tr_T_DeltaQ_Electron', 'B_Tr_T_DeltaQ_Proton', 'B_Tr_T_DeltaQ_Kaon', 'B_Tr_T_Signal_TagPart_PT', 'B_Tr_T_OWNPVIPSig', 'logEVIP', 'logP_proj', 'B_Tr_T_atanPT_PZ']
    load_extra = ['EVENTNUMBER','RUNNUMBER']
    features_noMC = [
        'B_OWNPV_X',
        'B_OWNPV_Y',
        'B_OWNPV_Z',
        'B_ENDV_X',
        'B_ENDV_Y',
        'B_ENDV_Z',
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
        #'B_Tr_T_firstX',
        #'B_Tr_T_firstY',
        #'B_Tr_T_firstZ',
        #'B_Tr_T_firstTX',
        #'B_Tr_T_firstTY',
        #'B_Tr_T_OWNPV_X',
        #'B_Tr_T_OWNPV_XERR',
        #'B_Tr_T_OWNPV_Y',
        #'B_Tr_T_OWNPV_YERR',
        #'B_Tr_T_OWNPV_Z',
        #'B_Tr_T_OWNPV_ZERR',
        #'B_Tr_T_Phi',
        #'B_Tr_T_M',
        'B_Tr_T_CHI2DOF',
        'B_Tr_T_GHOSTPROB',
        'B_Tr_T_PX',
        'B_Tr_T_PY',
        'B_Tr_T_PZ',
        'B_Tr_T_X',
        'B_Tr_T_Y',
        'B_Tr_T_Z',
        #'B_Tr_T_OBJECT_KEY',
        'B_Tr_T_IPChi2BVTX',
        'B_Tr_T_IPBVTX',]

    features_added += 'B_Tr_T_endSV_Z' # To test if this variable increases seperability of notSamePV and other classes
        
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
    features = features_noMC + features_added
    mc_info = ["B_BKGCAT", "B_Tr_T_absID", "B_Tr_T_Origin_Flag", "B_ID", "B_Tr_T_MC_MOTHER_ID", 
               'B_Tr_T_MC_MOTHER_KEY', 'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_KEY', 
               'B_Tr_T_MC_GD_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_KEY']

    loading_variables = features + mc_info + load_extra
 
    folders = ['Bd2JpsiKst', 'Bs2DsPi', 'Bu2JpsiK']

    # Iterate over each folder and collect the root files
    print(f"Loading data: Start \n", flush=True)

    df = pd.DataFrame(columns=loading_variables)

    for f in cfg.input_files:
        decay = os.path.basename(os.path.dirname(f))
        print(f"Reading input file: {f}", flush=True)
        with uproot.open("{}".format(f)) as _f:
            _df = _f[cfg.treename].arrays(loading_variables, library="pd")#[:500]#[:1_000_000]  # Limit the number of rows for testing TODO REMOVE
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

    condition_particle_pairs = [
    ((df.B_Tr_T_absID == 321)  & (df.B_Tr_T_Origin_Flag == 2),                                                                         "OSKaon"),
    ((df.B_Tr_T_absID == 13)   & (df.B_Tr_T_Origin_Flag == 2),                                                                         "OSMuon"),
    ((df.B_Tr_T_absID == 11)   & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) != 22 ),                                  "OSElectron"),
    ((df.B_Tr_T_absID == 211)  & (df.B_Tr_T_Origin_Flag == 1),                                                                         "SSPion"),
    ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1),                                                                         "SSProton"),
    ((df.B_Tr_T_absID == 321)  & (df.B_Tr_T_Origin_Flag == 1),                                                                         "SSKaon"),
    ((df.B_Tr_T_absID == 321)  & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag       != 100),                                  "otherK"),
    ((df.B_Tr_T_absID == 13)   & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag       != 100),                                  "otherMu"),
    ((df.B_Tr_T_absID == 11)   & (df.B_Tr_T_Origin_Flag != 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) != 22 ) & (df.B_Tr_T_Origin_Flag != 100), "otherE"),
    ((df.B_Tr_T_absID == 11)   & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) == 22 ),                                  "photonOSEl"),
    ((df.B_Tr_T_absID == 211)  & (df.B_Tr_T_Origin_Flag != 1) & (df.B_Tr_T_Origin_Flag       != 100),                                  "otherPi"),
    ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag != 1) & (df.B_Tr_T_Origin_Flag       != 100),                                  "otherP"),
    
    ((df.B_Tr_T_Origin_Flag == 100), "notSamePV"),
    ]

    train_particle_types = ["OSKaon", "OSMuon", "OSElectron", "SSPion", "SSProton", "SSKaon", "notSamePV"]
    

    if cfg.unify_SS:
        #combine the SSKaon and SSProton classes into a single class "SSKaon+SSProton"
        
        SSKaon_condition = None
        SSProton_condition = None
        for condition, particle in condition_particle_pairs:
            if particle == "SSKaon":
                SSKaon_condition = condition
            elif particle == "SSProton":
                SSProton_condition = condition

        condition_particle_pairs = [pair for pair in condition_particle_pairs if pair[1] not in ["SSKaon", "SSProton"]]
        condition_particle_pairs.append(((SSKaon_condition | SSProton_condition), "SSKaon+SSProton"))


    # Separate conditions and particle types for np.select()
    conditions = [pair[0] for pair in condition_particle_pairs]
    particle_type = [pair[1] for pair in condition_particle_pairs]

    # Assign particle types based on conditions, with default "Others" for unmatched rows
    df['particle'] = np.select(conditions, particle_type, default="Others")
    print(f"Number of tracks for each particle type:\n{df['particle'].value_counts()}", flush=True)

    #Save the dataframe with ids, flags and particles types to a root file for exploration
    with uproot.recreate(f"{cfg.target_path}/ids_flags_particles.root") as file:
        file["DecayTree"] = df[mc_info + ['particle']]
    
    df_export = df.copy()
    # convert all columns, except "particle", to float for better readability in c++
    # For string columns translate to integers first
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
                     "Others": 12,
                     "notSamePV": 13,}
    decay_dict = {"Bd2JpsiKst": 0, 
                  "Bs2DsPi": 1, 
                  "Bu2JpsiK": 2}
    print(df_export['particle'].unique())
    df_export['particle'] = df_export['particle'].map(particle_dict)
    df_export['decay'] = df_export['decay'].map(decay_dict)

    df_export = df_export.astype(float)
    print(df_export['particle'].unique())
    df_export['particle'] = df_export['particle'].astype(int)

    with uproot.recreate(f"{cfg.target_path}/DT_trainingset.root") as file:
        file["DecayTree"] = df_export
    del df_export

    print('Exploration dataframe with particle types, IDs and flags saved to root file', flush=True)


    print(f"Total number of tracks after removing 'Others': {df.shape[0]}", flush=True)
 

    print(f"Total number of tracks for each B candidate (by TRUE_ID):\n{df['B_ID'].value_counts()}", flush=True)

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

    print(f"\nComposition (%) before splitting in training-test set:\n{round(df.particle.value_counts()/df.shape[0],4)*100}", flush=True)
    print(f"\nComposition before splitting in training-test set:\n{df.particle.value_counts()}", flush=True)
    
    if downsampling:
        # Downsample the 'notSamePV', 'Others' classes. 
        # Get the count of the largest class excluding "notSamePV"
        max_class_size = df[df.particle == 'otherPi'].particle.value_counts().max()
        # Filter the 'notSamePV' rows
        not_same_pv_rows = df[df.particle == 'notSamePV']
        #others_rows = df[df.particle == 'Others']
        # Randomly sample the maximum class size from 'notSamePV'
        sampled_not_same_pv = not_same_pv_rows.sample(n=max_class_size, random_state=42)
        #others_rows = others_rows.sample(n=max_class_size, random_state=42)
        # Filter out 'notSamePV' from the original dataframe to keep the other rows
        #df = df[(df.particle != 'notSamePV') & (df.particle != 'Others')]
        df = df[(df.particle != 'notSamePV')]
        # Concatenate the sampled 'notSamePV' rows back with the other classes
        df = pd.concat([df, sampled_not_same_pv])
        #df = pd.concat([df, others_rows])
    # Optionally, shuffle the dataframe (to mix rows)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    x = df[features + ["particle"]].copy()
    y = x["particle"].copy()

    print(f'The features used are {len(features)}: {features}', flush=True)
    print('-----------------------------------------', flush=True)
    # To get same amount of not_taggingPart
    x.drop(columns=["particle"] , inplace = True)
    x_train , x_test ,y_train, y_test= train_test_split(x, y, test_size = 0.01, random_state=42)
    print(type(x_train))
    # Drop 'Other' particles from training dataset
    #Drop all particles that are not in train_particle_types, other particles are included for monitoring during testing
    mask = np.isin(y_train, train_particle_types)

    x_train = x_train[mask]
    y_train = y_train[mask]

    print(y_train)

    print(f"Classes used in training set: {y_train.unique()}", flush=True)

    print(f"Number of tracks in training set: {x_train.shape[0]}", flush=True)
    print(f"Composition of training set:\n{y_train.value_counts()}", flush=True)

    print(f"Number of tracks in test set: {x_test.shape[0]}", flush=True)
    print(f"Composition of test set:\n{y_test.value_counts()}", flush=True)
    
    if cfg.balanced == 'unbalanced':
        weights = None
    else:
        weights = str(cfg.balanced)
 
    if cfg.load:
        print("Loading the model...", flush=True)
        start_load = time.time()
        with open(f"{output_path}/decision_tree_model.pkl", "rb") as f:
            clf = pickle.load(f)
        print(f"Model loaded successfully! Type of clf: {type(clf)}", flush=True)
        print(f'Loading the Decision Tree required: {round(time.time()-start_load, 2)}s', flush=True)

    else:
        # Plot features
        if cfg.all_plots:
            print("Plotting features...", flush=True)
            DT_utils.plot_features_byOrigin(df, features, target_path=cfg.target_path, nbins=50)   

        print("Start fitting", flush=True)
        start_fit = time.time()
        clf = tree.DecisionTreeClassifier(max_depth = 6,class_weight=weights, min_impurity_decrease=0.009)
        clf.fit(x_train, y_train)
        print(f'Decision Tree training required: {round(time.time()-start_fit, 2)}s', flush=True)
        os.makedirs(output_path, exist_ok=True)

        # Save to a .pkl file
        with open(f"{output_path}/decision_tree_model.pkl", "wb") as f:
            pickle.dump(clf, f)
            
        
        # Get the text representation of the tree
        tree_rules = export_text(clf, feature_names=features)
        # Save the rules into a text file
        with open(f"{output_path}/decision_tree_rules.txt", "w") as file:
            file.write(tree_rules)

        # Visualize the decision tree
        dot_data = tree.export_graphviz(clf,feature_names=features,class_names=clf.classes_,filled=True, rounded=True, special_characters=True, proportion=True) 
        graph = graphviz.Source(dot_data) 
        graph.render(f"{output_path}/tree_schema")
        print("Model saved successfully!", flush=True)
    # Get all decision paths from the classifier
    paths = DT_utils.get_decision_paths(clf, features)

    # Group the paths by class label
    paths_by_class = defaultdict(list)
    for conditions, label in paths:
        # Combine conditions using AND for a single path.
        combined = " & ".join(conditions)
        paths_by_class[label].append(combined)

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
    DT_utils.plot_used_features(df, features_DT_used, target_path=cfg.target_path, nbins=50)
    print(f"Features used by the Decision Tree:\n {features_DT_used}", flush=True)




    # Write the conditions for each class into separate files.
    os.makedirs(f'{output_path}/cuts', exist_ok=True)
    for label, conditions_list in paths_by_class.items():
        # Create a file name based on the class label.
        filename = os.path.join(f'{output_path}/cuts', f"{label}_preselections.txt")
        with open(filename, "w") as f:
            # If multiple paths lead to the same class, separate them with OR.
            f.write("\nOR\n".join(conditions_list))
        print(f"Saved cuts for class '{label}' in {filename}", flush=True)
    
    print("Metrics for particle type composition: true VS predicted\n", flush=True)
    unify_classes = ["otherK", "otherMu", "otherE", "photonOSEl", "otherPi", "otherP"]
    y_pred_test = clf.predict(x_test)
    
    #Define order of particles in table/heatmap
    ordered_particles = ['OSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton', 'SSKaon','notSamePV', 'otherK', 'otherMu', 'otherE', 'photonOSEl', 'otherPi', 'otherP', 'Others']
    #Add in any potenially missing particles in the dataset (e.g. if some particle types are not present in the ordererd list but are in the dataset)
    for p in y_test.unique():
        if p not in ordered_particles:
            ordered_particles.append(p)


    DT_utils.metric_table(y_true=y_test, y_predicted=y_pred_test, possible_particle=ordered_particles, title='Versus True (pruned)', savepath=f"{output_path}/unified_pruned_confusion_normalised_by_truth.txt", unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_test, y_predicted=y_pred_test, possible_particle=ordered_particles, normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/unified_pruned_confusion_normalised_by_prediction.txt", unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_test, y_predicted=y_pred_test, possible_particle=ordered_particles, title='Versus True (pruned)', savepath=f"{output_path}/pruned_confusion_normalised_by_truth.txt", unify_classes=None)
    DT_utils.metric_table(y_true=y_test, y_predicted=y_pred_test, possible_particle=ordered_particles, normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/pruned_confusion_normalised_by_prediction.txt", unify_classes=None)

    
    print(f'Running the script required: {time.time()-start}s', flush=True)

    # Compute feature importance
    # print(f"Feature importance:\n", flush=True)
    # feat_import = clf.tree_.compute_feature_importances(normalize=True)
    # feat_import.sort()
    # for i in range(len(feat_import)):
    #     print(features[i],round(100*feat_import[i],2), flush=True)
    # print('-----------------------------------------', flush=True)
    # print(f"Permutation importance:\n", flush=True)
    # perm_import = clf.tree_.compute_feature_importances(normalize=True)
    # perm_import.sort()
    # for i in range(len(perm_import)):
    #     print(features[i],round(100*perm_import[i],2), flush=True)
    # print('-----------------------------------------', flush=True)