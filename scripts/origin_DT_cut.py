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
import configParameters as config
import glob
import argparse
# Local import
from IPython import embed
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
plt.rcParams['text.usetex'] = False # HD cluster has some problems with dvp not found
plt.rcParams.update({'axes.unicode_minus' : False})


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

def plot_features_byOrigin(data, features, particle_type, output_dir, nbins=100):
    # Plot input features 
    plt.figure(figsize=(40,40))
    
    for i, col in enumerate(data.columns.to_list()[:len(features)]):
        plt.subplot(3, 2, i + 1)
        plt.hist(data[col][data['ID_type']==particle_type['OSKaon']], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=ranges[col])
        plt.hist(data[col][data['ID_type']==particle_type['OSMuon']], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=ranges[col])
        plt.hist(data[col][data['ID_type']==particle_type['OSElectron']], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=ranges[col])
        plt.hist(data[col][data['ID_type']==particle_type['SSPion']], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=ranges[col])
        plt.hist(data[col][data['ID_type']==particle_type['SSProton']], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=ranges[col])
        plt.hist(data[col][data['ID_type']==particle_type['SSKaon']], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=ranges[col])
        #plt.hist(data[col][data['ID_type']==6], density = True, bins=nbins, label = f"{particle_type[6]}", histtype='step', color='r', )
        plt.xlabel(nice_names[col])
        plt.legend() 
        plt.tight_layout()
    if not os.path.exists(f"{cfg.target_path}/{output_dir}"):
        os.makedirs(f"{cfg.target_path}/{output_dir}")
    plt.savefig(f"{cfg.target_path}/{output_dir}/DT_features_byOrigin.pdf")

def plot_features_byParticle(data, features, output_dir, nbins=100):
   
    plt.figure(figsize=(40,40))
    for i, col in enumerate(data.columns.to_list()[:len(features)]):
        plt.subplot(3, 2, i + 1)
        plt.hist(data[col][data.B_Tr_T_absID==321], density = True, bins=nbins, label = f"Kaon", histtype='step', color='m', lw=2, range=ranges[col])
        plt.hist(data[col][data.B_Tr_T_absID==13], density = True, bins=nbins, label = f"Muon", histtype='step', color='b', lw=2, range=ranges[col])
        plt.hist(data[col][data.B_Tr_T_absID==11], density = True, bins=nbins, label = f"Electron", histtype='step', color='c', lw=2, range=ranges[col])
        plt.hist(data[col][data.B_Tr_T_absID==211], density = True, bins=nbins, label = f"Pion", histtype='step', color='g', lw=2, range=ranges[col])
        plt.hist(data[col][data.B_Tr_T_absID==2212], density = True, bins=nbins, label = f"Proton", histtype='step', color='y', lw=2, range=ranges[col])
        #plt.hist(data[col][data['ID_type']==6], density = True, bins=nbins, label = f"{particle_type[6]}", histtype='step', color='r', )
        plt.xlabel(nice_names[col])
        plt.legend() 
        plt.tight_layout()
    plt.savefig(f"{cfg.target_path}/{output_dir}/DT_features_byParticle.pdf")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='/ceph/users/molocco/classical-taggers/Data/withUT_MC_2024')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')

    cfg = parser.parse_args()

    from pprint import pprint
    pprint(cfg)
    start = time.time()
    run_time = time.time()
    features = ['B_Tr_T_PIDK', 'B_Tr_T_PIDe', 'B_Tr_T_PIDmu', 'B_Tr_T_PIDP', 'B_Tr_T_BVIPSig'] # 'B_Tr_T_ISMUON'?
    loading_variables = features+["B_Tr_T_absID", "B_Tr_T_Origin_Flag"]

    # Path to input root files
    file_pattern = f'/ceph/users/molocco/classical-taggers/Data/{config.sample_type}/2_added_features/*/*.root'
    input_paths = glob.glob(file_pattern)
    #input_paths = [path + ':Tuple/DecayTree' for path in input_paths] 

    print(f"Loading data: Start \n")
    df = pd.DataFrame(columns=loading_variables)
    for f in input_paths:
        print(f"Reading input file: {f}")
        with uproot.open("{}".format(f)) as _f:
            _df = _f[cfg.treename].arrays(loading_variables, library="pd")
        df = pd.concat([df, _df], ignore_index = True)

    print(f"Loading data finished in {round(-start+ time.time() , 2)}s")

    # with uproot.recreate(f'{outputPath}') as f:
    #     f['DecayTree'] = df
    # print(f'NTuple for Decision Tree saved at {outputPath}')

    df.B_Tr_T_Origin_Flag.astype(int)

    # Define labels for multiclassification
    conditions = [
    (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==2), # OSKaon
    (df.B_Tr_T_absID==13) & (df.B_Tr_T_Origin_Flag==2), # OSMuon
    (df.B_Tr_T_absID==11) & (df.B_Tr_T_Origin_Flag==2), # OSElectron
    (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1), # SSPion
    (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1), # SSProton
    (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1), # SSKaon
    ]
    particle_type = {"OSKaon":1,
                    "OSMuon":2,
                    "OSElectron":3,
                    "SSPion":4,
                    "SSProton":5,
                    "SSKaon":6}
    df['ID_type'] = np.select(conditions, particle_type.values())
    df.loc[~df['ID_type'].isin(particle_type.values()), 'ID_type'] = 0
    #Append IDs for particles that are nbot tagging particles
    #IDs.insert(0, 0)
    #particle_type.insert(0, 'not_taggingPart')

    # Plot features
    output_dir = 'DT_outputs'
    plot_features_byOrigin(data=df, features=features, particle_type=particle_type, output_dir=output_dir)
    plot_features_byParticle(data=df, features=features, output_dir=output_dir)

    # Shuffle 
    df = df.sample(frac=1)
    df.dropna(inplace=True)
    x = df.loc[df.ID_type != 0][features + ["ID_type"]]
    print(f"\nComposition:\n{round(x.ID_type.value_counts()/x.shape[0],4)*100}")
    print('-----------------------------------------')
    # To get same amount of not_taggingPart
    #x = pd.concat([x, df.loc[df.ID_type == 0][features + ["ID_type"]].head(len(x))])
    y = x.ID_type
    x.drop(columns="ID_type" , inplace = True)
    x_train , x_test ,y_train, y_test= train_test_split(x, y, test_size = 0.3, random_state=42)

    start = time.time()
    print("Start fitting")
    clf = tree.DecisionTreeClassifier(max_depth = 4, class_weight='balanced')

    #embed()
    clf.fit(x_train, y_train)
    print(f"Fit in: {round(-start+ time.time() , 2)}s\n")

    # Compute feature importance
    print(f"Feature importance:\n")
    feat_import = clf.tree_.compute_feature_importances(normalize=True)
    feat_import.sort()
    for i in range(len(feat_import)):
        print(features[i],round(100*feat_import[i],2))
    print('-----------------------------------------')


    '''   
    #Plot the ROC Curve
    y_test_predict = clf.predict_proba(x_test)[:,1]
    y_train_predict = clf.predict_proba(x_train)[:,1]
    plt.figure()
    fpr_test, tpr_test,_ = roc_curve(y_test,y_test_predict)
    roc_auc_test = round(auc(fpr_test, tpr_test),2)
    fpr_train, tpr_train,_ = roc_curve(y_train,y_train_predict)
    roc_auc_train = round(auc(fpr_train, tpr_train),2)
    lw  = 2
    plt.plot(fpr_test, tpr_test, color='darkorange',
        lw=lw, label=f'Test(area = {roc_auc_test})' )
    plt.plot(fpr_train, tpr_train, color='darkblue',
        lw=lw, label=f'Train(area = {roc_auc_train})' )
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve ')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/ ROC_AUC.pdf")
    #plt.show()
    plt.close()
    '''

    # Visualize the decision tree
    dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True,special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(f"{output_dir}/tree_schema")

    #print(f"Accuracy:{clf.score(x_test,y_test)}")
