import glob

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
from collections import defaultdict
from sklearn.tree import export_text
import re

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
    parser.add_argument('--BKG0', help='If specified, only BGKCAT=0 tracks are used',  action='store_true') # Per default BKG0==0 are removed
    parser.add_argument('--load', help='If specified, DT is loaded, instead of trained',  action='store_true')
    parser.add_argument('--all_plots', help='If specified, a histogramm of all variables is plotted',  action='store_true') 
    parser.add_argument('--downsample', help='If specified, the notSamePV class is downsampled to have the same number of tracks as the largest other class (excluding notSamePV)',  action='store_true')
    parser.add_argument('--train_classes', help='Particle types included in the training of the DT, other particle classes are included in the test set for monitoring.', default=["OSKaon", "OSMuon", "OSElectron", "SSPion", "SSProton", "SSKaon", 'notSamePV'], nargs='+', type=str)
    parser.add_argument('--save_dataframes', help='If specified, dataframes containing training and exporatory information is saved to disk for debugging or prototyping',  action='store_true')
    parser.add_argument('--conf_weight_config', help='A dictionary containing the parameter constructing the confusion matrix weights.', type=str,)
    parser.add_argument('--lda_classes', help='Particle types included in the LDA training. If none no LDA is performed. LDA features are supplied to the DT.', default=None, nargs='+', type=str)
    parser.add_argument('--feature_set', help='The set of features to use for training the DT.', default='v1_set', type=str)

    print(f'Run at time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    cfg = parser.parse_args()

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
    if cfg.unify_SS:
        output_path = f'{cfg.target_path}/SSKSSP/'
    else:
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
            _df = _f[cfg.treename].arrays(loading_variables, library="pd")#[:1_000_000]  # Limit the number of rows for testing TODO REMOVE
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
                    "Others": 13,}
    decay_dict = {"Bd2JpsiKst": 0, 
                "Bs2DsPi": 1, 
                "Bu2JpsiK": 2}
    #Save the dataframe with ids, flags and particles types to a root file for exploration
    if cfg.save_dataframes:
        with uproot.recreate(f"{cfg.target_path}/ids_flags_particles.root") as file:
            file["DecayTree"] = df[mc_info + ['particle']]
        
        df_export = df.copy()
        # convert all columns, except "particle", to float for better readability in c++
        # For string columns translate to integers first
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

    print(f"\nComposition (%) before splitting in training-test set:\n{round(df.particle.value_counts()/df.shape[0],4)*100}", flush=True)
    print(f"\nComposition before splitting in training-test set:\n{df.particle.value_counts()}", flush=True)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    x = df[features + ["particle"]].copy()
    y = x["particle"].copy()

    print(f'The features used are {len(features)}: {features}', flush=True)
    print('-----------------------------------------', flush=True)
    # To get same amount of not_taggingPart
    x.drop(columns=["particle"] , inplace = True)
    X_train , x_test ,y_train, y_test= train_test_split(x, y, test_size = 0.01, random_state=42)
    print(type(X_train))
    # Drop 'Other' particles from training dataset
    #Drop all particles that are not in train_particle_types, other particles are included for monitoring during testing
    mask = np.isin(y_train, cfg.train_classes)

    X_train = X_train[mask]
    y_train = y_train[mask]

    if cfg.downsample:
        # Downsample all non tagging particle classes (this case only notSamePV, if other background classes are added in the future add them here) 
        # to have the same number of tracks as the largest tagging particle class (which is SSPion)
        # Apply downsampling to the training set only.

        # Downsample the 'notSamePV' classes. 
        # Get the count of the largest class excluding "notSamePV"
        X_train["particle"] = y_train

        # max_class_size = X_train[X_train["particle"] == 'SSPion'].value_counts().max()
        max_class_size = X_train.loc[X_train["particle"] != 'notSamePV', "particle"].value_counts().max()

        # Filter the 'notSamePV' rows
        not_same_pv_rows = X_train[X_train["particle"] == 'notSamePV']
        # Randomly sample the maximum class size from 'notSamePV'
        sampled_not_same_pv = not_same_pv_rows.sample(n=max_class_size, random_state=42)
        # Filter out 'notSamePV' from the original dataframe to keep the other rows
        X_train = X_train.loc[(X_train["particle"] != 'notSamePV')]
        # Concatenate the sampled 'notSamePV' rows back with the other classes
        X_train = pd.concat([X_train, sampled_not_same_pv])

        y_train = X_train["particle"]
        X_train.drop(columns=["particle"], inplace=True)

        #df = pd.concat([df, others_rows])
        print(f'Composition after downsampling:\n{y_train.value_counts()}', flush=True)

        del not_same_pv_rows, sampled_not_same_pv

    
    if cfg.lda_classes:
        # Perform LDA on the specified classes and add the LDA features to the training and test sets
        print(f"Performing LDA on classes: {cfg.lda_classes}", flush=True)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(x_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=x_test.columns, index=x_test.index)

        X_train_scaled['particle'] = y_train
        
        # sample 10% of the training data for LDA and drop it from the trainingset of the DT
        X_train_lda = X_train_scaled.sample(frac=0.1, random_state=42)


        X_train_lda = X_train_lda[X_train_lda['particle'].isin(cfg.lda_classes)]

        X_train = X_train.drop(X_train_lda.index)
        y_train = X_train_scaled.drop(X_train_lda.index)["particle"]
        X_train_scaled.drop(columns=["particle"], inplace=True)
        X_train_scaled.drop(X_train_lda.index, inplace=True)
        

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_lda.drop(columns=["particle"]), X_train_lda["particle"])


        X_train_transformed = lda.transform(X_train_scaled)    
        X_test_transformed = lda.transform(X_test_scaled)

        with open(f"{output_path}/lda_scaler.pkl", "wb") as f:
            dump(scaler, f)

        with open(f"{output_path}/lda_model.pkl", "wb") as f:
            dump(lda, f)

        
        lda_cols = [f"LDA_{i}" for i in range(X_train_transformed.shape[1])]
        

        X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=lda_cols, index=X_train.index)
        X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=lda_cols, index=x_test.index)

        print(f"Transformed features:\n{X_train_transformed_df.head()}")

        X_train = pd.merge(X_train, X_train_transformed_df, left_index=True, right_index=True)
        x_test = pd.merge(x_test, X_test_transformed_df, left_index=True, right_index=True)


        colors = {"OSKaon": 'red', "OSMuon": 'blue', "OSElectron": 'green', "SSPion": 'orange', "SSProton": 'purple', "SSKaon": 'cyan', "notSamePV": 'black'}
        # label = {0: "OSKaon", 1: "OSMuon", 2: "OSElectron", 3: "SSPion", 4: "SSProton", 5: "SSKaon", 12: "notSamePV"}


        mask = y_test.isin(cfg.lda_classes)
        y_test_plot = y_test[mask]

        X_test_plot = X_test_transformed_df[mask]

        X_test_plot['particle'] = y_test_plot
        X_test_plot['color'] = y_test_plot.map(colors)        





        plot_features = lda_cols
        n_bins = 50
        fig, axes = plt.subplots(len(plot_features), len(plot_features), figsize=(len(plot_features)*3,len(plot_features)*3))
        for i, feature1 in enumerate(plot_features):
            for j, feature2 in enumerate(plot_features):
                ax = axes[i, j]
                if i == j:
                    min_val = X_test_plot[feature1].min()
                    max_val = X_test_plot[feature1].max()
                    bins = np.linspace(min_val, max_val, n_bins)

                    for particle in cfg.lda_classes:
                        subset = X_test_plot[X_test_plot['particle'] == particle]
                        ax.hist(subset[feature1], bins=bins, alpha=0.5, label=particle, color=colors[particle], density=True, )
                    # ax.hist(df_test_transformed[feature1], bins=bins, color='gray', alpha=0.7)
                    ax.set_xlabel(feature1)
                    ax.set_ylabel(f'Density per {np.round((max_val - min_val)/n_bins, 2)}')
                    if j == 0:
                        ax.set_xlim(min_val-0.4, max_val)
                        ax.legend(loc='best')

                else:
                    scatter = ax.scatter(X_test_plot[feature2], X_test_plot[feature1], c=X_test_plot['color'].values, cmap='viridis', alpha=0.5, s=1)
                    ax.set_xlabel(feature2)
                    ax.set_ylabel(feature1)
        # add a legend for the colors
        handles, labels = scatter.legend_elements()

        del X_test_plot, y_test_plot

        plt.tight_layout()
        plt.savefig(f"{cfg.target_path}/LDA_feature_scatter_matrix.png", dpi=300)



        scalings_matrix = lda.scalings_
        feature_labels = features

        plt.figure(figsize=(4, 18))
        sns.heatmap(
            scalings_matrix,
            xticklabels=lda_cols,
            yticklabels=feature_labels,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Scaling Coefficient'}
        )
        plt.title('LDA Scalings Matrix Heatmap')
        plt.xlabel('Linear Discriminants')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(f"{cfg.target_path}/LDA_scalings_matrix.png", dpi=300)

        if cfg.save_dataframes:
            with uproot.recreate(f"{cfg.target_path}/Data_lda_features.root") as file:
                file["DecayTree"] = pd.concat([X_train_transformed_df, X_test_transformed_df], axis=0)

        del X_train_transformed_df

        print(f"Training set after adding LDA features:\n{X_train.head()}")
        print(f"Columns after adding LDA features:\n{X_train.columns.tolist()}")



    print(y_train)

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
            DT_utils.plot_features_byOrigin(df, features, target_path=cfg.target_path, nbins=50)   

        print("Start fitting", flush=True)
        start_fit = time.time()
        # clf = tree.DecisionTreeClassifier(max_depth = 6,class_weight=weights, min_impurity_decrease=0.009)
        # clf.fit(X_train, y_train)
        
        particle_labels = particle_type+['others']
        X_train_val = X_train.to_numpy(copy=False)
        X_train_val = X_train_val.astype(np.float32)
        
        y_train_val = y_train.to_numpy()
        # Change y_train_val to in32 for c++ compatibility, since pyDecisionTree expects the target to be of type int32
        # use the particle_dict to convert the particle types to integers
        y_train_val = np.array([particle_dict[particle] for particle in y_train_val])
        y_train_val = y_train_val.astype(np.int32)

        cpp_df = pyDecisionTree.Dataframe(X_train_val, y_train_val, X_train.columns.to_list(), particle_labels, debugInfo=False)

        if cfg.balanced == 'unbalanced':
            balancing_weights = None
        else:
            balancing_weights = cpp_df.get_balancing_weights()

        print(f"Balancing weights: {balancing_weights}", flush=True)

        if conf_weights is None:
            criterion = pyDecisionTree.Gini()
        else:
            criterion = pyDecisionTree.Gini_conf_weighted(conf_weights)

        clf = pyDecisionTree.DecisionTree(criterion, 6, X_train.columns.to_list(), particle_labels, 2, 1, 0.0, 0.009, verbose=False)
        clf.fit(cpp_df.get_data(), cpp_df.get_targets(), balancing_weights)



        print(f'Decision Tree training required: {round(time.time()-start_fit, 2)}s', flush=True)
        os.makedirs(output_path, exist_ok=True)

        clf.save_tree(f"{output_path}/decision_tree_model.yaml")
        # Copy the tree to a txt file as well. Snakemake expects a txt file for the decision tree which is deleted anytime the snakemake rule is run.
        # The yaml remains allowing to load the model without retraining it.
        shutil.copyfile(f"{output_path}/decision_tree_model.yaml", f"{output_path}/decision_tree_model.txt")
        clf.plot_tree(f"{output_path}/tree_schema.pdf")

        print("Model saved successfully!", flush=True)
    # Get all decision paths from the classifier

    clf.export_cuts(os.path.join(output_path, "cuts/"))


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

    print(paths_by_class)

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



    print("Metrics for particle type composition: true VS predicted\n", flush=True)
    unify_classes = ["otherK", "otherMu", "otherE", "photonOSEl", "otherPi", "otherP"]

    print('Beginning prediction on test set...', flush=True)
    y_pred_test = clf.predict(x_test)

    inv_particle_dict = {v: k for k, v in particle_dict.items()}
    y_pred_test = np.array([inv_particle_dict[pred] for pred in y_pred_test])
    y_test = y_test.to_numpy()

    print(f"Prediction on test set completed. Number of predictions: {len(y_pred_test)}", flush=True)
    print(f"Unique predicted classes: {np.unique(y_pred_test)}", flush=True)
    print(f"Predicted classes: {y_pred_test}", flush=True)
    print(f"True classes: {y_test}", flush=True)

    #Define order of particles in table/heatmap
    ordered_particles = ['OSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton', 'SSKaon','notSamePV', 'otherK', 'otherMu', 'otherE', 'photonOSEl', 'otherPi', 'otherP', 'Others']
    #Add in any potenially missing particles in the dataset (e.g. if some particle types are not present in the ordererd list but are in the dataset)
    for p in np.unique(y_test):
        if p not in ordered_particles:
            ordered_particles.append(p)


    DT_utils.metric_table(y_true=y_test, y_predicted=y_pred_test, possible_particle=ordered_particles, title='Versus True (pruned)', savepath=f"{output_path}/unified_pruned_confusion_normalised_by_truth.txt", unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_test, y_predicted=y_pred_test, possible_particle=ordered_particles, normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/unified_pruned_confusion_normalised_by_prediction.txt", unify_classes=unify_classes)
    DT_utils.metric_table(y_true=y_test, y_predicted=y_pred_test, possible_particle=ordered_particles, title='Versus True (pruned)', savepath=f"{output_path}/pruned_confusion_normalised_by_truth.txt", unify_classes=None)
    DT_utils.metric_table(y_true=y_test, y_predicted=y_pred_test, possible_particle=ordered_particles, normalization='predicted', title='Versus Predicted (pruned / balanced)', savepath=f"{output_path}/pruned_confusion_normalised_by_prediction.txt", unify_classes=None)


    print(f'Running the script required: {time.time()-start}s', flush=True)
