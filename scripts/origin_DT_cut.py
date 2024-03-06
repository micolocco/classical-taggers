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
from IPython import embed
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

def plot_features(data, IDs, particle_type, output_dir, nbins=100):

    # Plot input features 
    plt.figure(figsize=(40,40))
    try:
        for i, col in enumerate(data.columns.to_list()[:5]):
            plt.subplot(3, 2, i + 1)
            plt.hist(data[col][data['ID_type']==0], density = True, bins=nbins, label = f"{particle_type[0]}", fill=True, color='m', alpha=0.5)
            plt.hist(data[col][data['ID_type']==1], density = True, bins=nbins, label = f"{particle_type[1]}", fill=True, color='b', alpha=0.5)
            plt.hist(data[col][data['ID_type']==2], density = True, bins=nbins, label = f"{particle_type[2]}", fill=True, color='c', alpha=0.5)
            plt.hist(data[col][data['ID_type']==3], density = True, bins=nbins, label = f"{particle_type[3]}", fill=True, color='g', alpha=0.5)
            plt.hist(data[col][data['ID_type']==4], density = True, bins=nbins, label = f"{particle_type[4]}", fill=True, color='y', alpha=0.5)
            plt.hist(data[col][data['ID_type']==5], density = True, bins=nbins, label = f"{particle_type[5]}", fill=True, color='r', alpha=0.5)
            plt.hist(data[col][data['ID_type']==6], density = True, bins=nbins, label = f"{particle_type[6]}", fill=True, color='r', alpha=0.5)

            plt.legend()
            plt.title(col)
            plt.tight_layout()
        plt.savefig(f"{output_dir}/DT_features.pdf")
    except Exception as e:
        print(col,e)



start = time.time()
run_time = time.time()
features = ['B_Tr_T_PIDK', 'B_Tr_T_PIDe', 'B_Tr_T_PIDmu', 'B_Tr_T_PIDP', 'B_Tr_T_BVIPSig', 'B_Tr_T_ISMUON']

# Path to input root file
#file_pattern = f'/eos/lhcb/user/m/miolocco/FT_NTuple/{config.sample_type}/2_added_features/*/notSelected.root:DecayTree'
#input_paths = glob.glob(file_pattern)
#input_paths = [path + ':Tuple/DecayTree;1' for path in input_paths] 

input_paths = '/eos/lhcb/user/m/miolocco/FT_NTuple/withUT_MC_2024/2_added_features/Bs2DsPi/notSelected.root'
outputPath = f'/eos/lhcb/user/m/miolocco/FT_NTuple/{config.sample_type}/2_added_features/allDecays_DTinput.root'
output_dir = 'DT_outputs'

loading_variables = features+["B_Tr_T_absID", "B_Tr_T_Origin_Flag"]

df_save = pd.DataFrame(columns=loading_variables) #Book dataframe for saving
print(f"Loading data: Start \n")

for df in uproot.iterate(f'{input_paths}:DecayTree', loading_variables, step_size=100000, library = 'pd'):
        df_save = pd.concat([df_save, df], ignore_index = True, copy = False)
        print(f"{df_save.shape[0]} tracks will be saved")
        print(f'Block finished in {round(time.time() - run_time,2)}s')
        print('--------------------End Block------------------------------')
        print()

print(f"Loading data finished in {round(-start+ time.time() , 2)}s")
print(f"Saved {df_save.shape[0]} tracks in dataframe")
with uproot.recreate(f'{outputPath}') as f:
    f['DecayTree'] = df_save
print(f'NTuple for Decision Tree saved at {outputPath}')

df.B_Tr_T_Origin_Flag.astype(int)

# Define labels for multiclassification
conditions = [
   (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==2),
   (df.B_Tr_T_absID==13) & (df.B_Tr_T_Origin_Flag==2),
   (df.B_Tr_T_absID==11) & (df.B_Tr_T_Origin_Flag==2),
   (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1),
   (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1),
   (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1),
]
IDs = [1, 2, 3, 4, 5, 6] 
particle_type = ["OSKaon", "OSMuon", "OSElectron", "SSPion", "SSProton", "SSKaon"]
df['ID_type'] = np.select(conditions, IDs)
df.loc[~df['ID_type'].isin(IDs), 'ID_type'] = 0
#Append IDs for particles that are nbot tagging particles
IDs.insert(0, 0)
particle_type.insert(0, 'not_taggingPart')

print(f"Composition:\n{round(df.ID_type.value_counts()/df.shape[0],4)*100}")

# Plot features
plot_features(df, IDs, particle_type, output_dir)

# Shuffle 
df = df.sample(frac=1)
df.dropna(inplace=True)
x = df.loc[df.ID_type != 0][features + ["ID_type"]]

# To get same amount of not_taggingPart
x = pd.concat([x, df.loc[df.ID_type == 0][features + ["ID_type"]].head(len(x))])
y = x.ID_type
x.drop(columns="ID_type" , inplace = True)

x_train , x_test ,y_train, y_test= train_test_split(x, y, test_size = 0.3, random_state=42)

start = time.time()
print("Start fitting")
clf = tree.DecisionTreeClassifier(max_depth = 5, class_weight='balanced')

embed()
clf.fit(x_train, y_train)
print(f"Fit in: {round(-start+ time.time() , 2)}s")

# If the Feature importance is needed
printFeatImport = True
if printFeatImport:
    feat_import = clf.tree_.compute_feature_importances(normalize=True)
    for i in range(len(feat_import)):
        print(features[i],round(100*feat_import[i],2))

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
dot_data = tree.export_graphviz(clf,feature_names=features,class_names=particle_type,filled=True, rounded=True,special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render(f"{output_dir}/tree_schema")

#print(f"Accuracy:{clf.score(x_test,y_test)}")
