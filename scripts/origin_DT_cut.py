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
import glob
import argparse
import datetime
from collections import defaultdict

# Local import
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
3 == OS Fragmentation from excited B
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Try a Decision Tree for selecting different tagging particle types',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    #base_pattern = '/ceph/users/molocco/Data/withUT_MC_2024/2_added_features'
    parser.add_argument('--base_pattern', help='Pattern for input files', type=str)
    parser.add_argument('--target_path', help='Name of the output dir', type=str,)
    parser.add_argument('--balanced', help='If classes are balanced or unbalanced', choices=('balanced', 'unbalanced'), type=str, )
    parser.add_argument('--unify_SS', help='If unify SSKaon and SSProton in a single class', action='store_true' ) # action='store_true' means args.unify_SS will be set to True if the --unify_SS argument is provided on the command line.
    # Per default BKG0==0 are removed
    parser.add_argument('--BKG0', help='If specified, only BGKCAT=0 tracks are used',  action='store_true') # action='store_true' means args.BKG0 will be set to True if the --0 argument is provided on the command line.
    parser.add_argument('--load', help='If specified, DT is loaded, or trained',  action='store_true') # action='store_true' means args.load will be set to True if the --load argument is provided on the command line.

    print(f'Run at time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

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

    features_added = ['B_Tr_T_minPhiDistance', 'B_Tr_T_cos_PhiDistance', 'B_Tr_T_PhiDistance', 'B_Tr_T_diff_z', 'B_Tr_T_DeltaR', 'diff_P', 'P_proj', 't', 'EVIP', 'B_Tr_T_absOWNPV_IP', 'B_Tr_T_EtaDistance', 'B_Tr_T_DeltaQ_Pion', 'B_Tr_T_DeltaQ_Muon', 'B_Tr_T_DeltaQ_Electron', 'B_Tr_T_DeltaQ_Proton', 'B_Tr_T_DeltaQ_Kaon', 'B_Tr_T_Signal_TagPart_PT', 'B_Tr_T_eoverP', 'B_Tr_T_OWNPVIPSig', 'logEVIP', 'logP_proj', 'B_Tr_T_atanPT_PZ']
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
    features = features_noMC + features_added
    mc_info = ["B_BKGCAT", "B_Tr_T_absID", "B_Tr_T_Origin_Flag", "B_TRUEID", "B_Tr_T_MC_MOTHER_ID"]    
    loading_variables = features + mc_info + load_extra
 
    folders = ['Bd2JpsiKst', 'Bs2DsPi', 'Bu2JpsiK']
    #folders = ['Bd2JpsiKst',  'Bu2JpsiK'] 
    # NEED TO INCLUDE BS!!!!!!!

    # Iterate over each folder and collect the root files
    print(f"Loading data: Start \n")

    df = pd.DataFrame(columns=loading_variables)
    
    for decay in folders:
        pattern = f'{cfg.base_pattern}/{decay}/*1_1.mc.root'
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
    print(f"Total number of tracks (all decays, all particles): {df.shape[0]}") 

    df['B_Tr_T_Origin_Flag'] = df['B_Tr_T_Origin_Flag'].astype(int)
    df['B_Tr_T_MC_MOTHER_ID'] = df['B_Tr_T_MC_MOTHER_ID'].astype(int)
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
        #((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 2), "OSProton"),
        ((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1), "SSKaon"),
        #((df.B_Tr_T_absID == 321) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag != 100), "otherK"),
        #((df.B_Tr_T_absID == 13) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag != 100), "otherMu"),
        #((df.B_Tr_T_absID == 11) & (df.B_Tr_T_Origin_Flag != 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) != 22) & (df.B_Tr_T_Origin_Flag != 100), "otherE"),
        #((df.B_Tr_T_absID == 11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) == 22), "photonOSEl"),
        #((df.B_Tr_T_absID == 211) & (df.B_Tr_T_Origin_Flag != 1) & (df.B_Tr_T_Origin_Flag != 100), "otherPi"),
        #((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag != 1)  & (df.B_Tr_T_Origin_Flag != 100), "otherP"),
        #((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag != 1) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag != 100), "noOSSSProton"),
        ((df.B_Tr_T_Origin_Flag == 100), "notSamePV"),
        ] #

    # Separate conditions and particle types for np.select()
    conditions = [pair[0] for pair in condition_particle_pairs]
    particle_type = [pair[1] for pair in condition_particle_pairs]

    # Assign particle types based on conditions, with default "Others" for unmatched rows
    df['particle'] = np.select(conditions, particle_type, default="Others")
    
    
    # To remove all the other particles
    df = df.loc[(df.particle != 'Others' )]

    print(f"Total number of tracks after removing 'Others': {df.shape[0]}")
 
    # For SS case if B_Tr_T_Charge has same sign of B_TRUE_ID is a correct tagging particle candidate
    # We want to remove all the SSKaon from Bd2JpsiKst and the SSPion, SSProton from Bs2DsPi
    # We want to remove all the SSKaon (or SSPion/SSProton) that will return a wrong tagging decision
    #df['sign_tag'] = (df['B_TRUEID']/abs(df['B_TRUEID'])) * df['B_Tr_T_Charge']
    print(f"Total number of tracks for each B candidate (by TRUE_ID):\n{df['B_TRUEID'].value_counts()}")
   # plot_features(df, features, )
    
    '''
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
    # Filter the DataFrame to remove rows that meet the combined condition
    print(f"Applying removal conditions")
    df_filtered = df[~removal_conditions].copy()
    '''
    df_filtered = df.copy()
    # Leaving it as an option, but only BKG_CAT==0 should be the default
    if cfg.BKG0:
        df_filtered['B_BKGCAT'] = df_filtered['B_BKGCAT'].astype(int)
        print("Filtering tracks based on decay and B_BKGCAT values...")

        mask = (
            (df_filtered['decay'].str.contains('Bs2DsPi') & (df_filtered['B_BKGCAT'] == 20)) # Only for Bs2DsPi due to problems with the BKG_CAT
            | (~df_filtered['decay'].str.contains('Bs2DsPi') & (df_filtered['B_BKGCAT'] == 0))
        )
        df_filtered = df_filtered.loc[mask]
        print(f"New number of tracks: {df_filtered.shape[0]}")
    else:
        DT_utils.count_BKGCAT(df_filtered)

    print(f"\nComposition (%) before splitting in training-test set:\n{round(df_filtered.particle.value_counts()/df_filtered.shape[0],4)*100}")
    print(f"\nComposition before splitting in training-test set:\n{df_filtered.particle.value_counts()}")
    
    if downsampling:
        # Downsample the 'notSamePV', 'Others' classes. 
        # Get the count of the largest class excluding "notSamePV"
        max_class_size = df_filtered[df_filtered.particle == 'otherPi'].particle.value_counts().max()
        # Filter the 'notSamePV' rows
        not_same_pv_rows = df_filtered[df_filtered.particle == 'notSamePV']
        #others_rows = df_filtered[df_filtered.particle == 'Others']
        # Randomly sample the maximum class size from 'notSamePV'
        sampled_not_same_pv = not_same_pv_rows.sample(n=max_class_size, random_state=42)
        #others_rows = others_rows.sample(n=max_class_size, random_state=42)
        # Filter out 'notSamePV' from the original dataframe to keep the other rows
        #df_filtered = df_filtered[(df_filtered.particle != 'notSamePV') & (df_filtered.particle != 'Others')]
        df_filtered = df_filtered[(df_filtered.particle != 'notSamePV')]
        # Concatenate the sampled 'notSamePV' rows back with the other classes
        df_filtered = pd.concat([df_filtered, sampled_not_same_pv])
        #df_filtered = pd.concat([df_filtered, others_rows])
    # Optionally, shuffle the dataframe (to mix rows)

    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

    x = df_filtered[features + ["particle"]].copy()
    y = x["particle"].copy()

    print(f'The features used are {len(features)}: {features}')
    print('-----------------------------------------')
    # To get same amount of not_taggingPart
    #x = pd.concat([x, df.loc[df.particle == 0][features + ["particle"]].head(len(x))])
    x.drop(columns=["particle"] , inplace = True)
    x_train , x_test ,y_train, y_test= train_test_split(x, y, test_size = 0.01, random_state=42)
    #print(f"Composition of the training sample:\n{round(x.particle.value_counts()/x.shape[0],4)*100}")  

    if cfg.balanced == 'unbalanced':
        weights = None
    else:
        weights = str(cfg.balanced)
 
    if cfg.load:
        print("Loading the model...")
        start_load = time.time()
        with open(f"{output_path}/decision_tree_model.pkl", "rb") as f:
            clf = pickle.load(f)
        print(f"Model loaded successfully! Type of clf: {type(clf)}")
        print(f'Loading the Decision Tree required: {round(time.time()-start_load, 2)}s')

    else:
        # Plot features
       # print("Plotting features...")
       # DT_utils.plot_features_byOrigin(df_filtered, features, particle_type, target_path=cfg.target_path, nbins=50)   
        print("Start fitting")
        start_fit = time.time()
        clf = tree.DecisionTreeClassifier(max_depth = 6,class_weight=weights, min_impurity_decrease=0.009)
        clf.fit(x_train, y_train)
        print(f'Decision Tree training required: {round(time.time()-start_fit, 2)}s')
        os.makedirs(output_path, exist_ok=True)

        # Save to a .pkl file
        with open(f"{output_path}/decision_tree_model.pkl", "wb") as f:
            pickle.dump(clf, f)
            
        from sklearn.tree import export_text
        # Get the text representation of the tree
        tree_rules = export_text(clf, feature_names=features)
        # Save the rules into a text file
        with open(f"{output_path}/decision_tree_rules.txt", "w") as file:
            file.write(tree_rules)

        # Visualize the decision tree
        #dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True,special_characters=True) 
        dot_data = tree.export_graphviz(clf,feature_names=features,class_names=clf.classes_,filled=True, rounded=True, special_characters=True, proportion=True) 
        graph = graphviz.Source(dot_data) 
        graph.render(f"{output_path}/tree_schema")
        print("Model saved successfully!")
    #print(f"Accuracy:{clf.score(x_test,y_test)}")
    # Get all decision paths from the classifier
    paths = DT_utils.get_decision_paths(clf, features)

    # Group the paths by class label
    paths_by_class = defaultdict(list)
    for conditions, label in paths:
        # Combine conditions using AND for a single path.
        combined = " & ".join(conditions)
        paths_by_class[label].append(combined)

    # Write the conditions for each class into separate files.
    os.makedirs(f'{output_path}/cuts', exist_ok=True)
    for label, conditions_list in paths_by_class.items():
        # Create a file name based on the class label.
        filename = os.path.join(f'{output_path}/cuts', f"{label}_preselections.txt")
        with open(filename, "w") as f:
            # If multiple paths lead to the same class, separate them with OR.
            f.write("\nOR\n".join(conditions_list))
        print(f"Saved cuts for class '{label}' in {filename}")
    
    print("Metrics for particle type composition: true VS predicted\n")
    unify_classes = ["otherK", "otherMu", "otherE", "photonOSEl", "otherPi", "otherP", "Others"]
    DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), possible_particle=sorted(df_filtered['particle'].unique()), title='Versus True (pruned)', savepath=f"{output_path}/pruned_confusion_normalised_by_truth.txt", unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), possible_particle=sorted(df_filtered['particle'].unique()), normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/pruned_confusion_normalised_by_prediction.txt", unify_classes=unify_classes)

    #DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), possible_particle=sorted(df_filtered['particle'].unique()), normalization='predicted', title='Versus Predicted (pruned)', savepath=f"{output_path}/pruned_confusion_normalised_by_prediction.txt")
    ##DT_utils.metric_table(y_true=y_train, y_predicted=clf.predict(x_train), possible_particle=sorted(df_filtered['particle'].unique()), balanced=True, title='Versus True (balanced)') # same as unbalanced
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