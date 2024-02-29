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

'''
Origin Flag values:

0 == Signal
1 == SS Fragmentation
2 == OS Decay (track has B0, B+, Bs, Bc+ mother)
3 == OS Fragmentationfrom excited B
4 == OS Frag from b quark
5 == Prompt
100 == Tracks from other vertex
-1 == No associated MC particle (probably ghost)

'''

def plot_features(data, values, nbins=100):

    # Plot input features 
    plt.figure(figsize=(24,50))
    try:
        for i, col in enumerate(data.columns.to_list()[:5]):
            plt.subplot(3, 2, i + 1)
            plt.hist(data[col][data['particle_type']==values[0]], density = True, bins=nbins, label = f"{values[0]}", fill=False, color='m', alpha=0.5)
            plt.hist(data[col][data['particle_type']==values[1]], density = True, bins=nbins, label = f"{values[1]}", fill=False, color='b', alpha=0.5)
            plt.hist(data[col][data['particle_type']==values[2]], density = True, bins=nbins, label = f"{values[2]}", fill=False, color='c', alpha=0.5)
            plt.hist(data[col][data['particle_type']==values[3]], density = True, bins=nbins, label = f"{values[3]}", fill=False, color='g', alpha=0.5)
            plt.hist(data[col][data['particle_type']==values[4]], density = True, bins=nbins, label = f"{values[4]}", fill=False, color='y', alpha=0.5)
            plt.hist(data[col][data['particle_type']==values[5]], density = True, bins=nbins, label = f"{values[5]}", fill=False, color='r', alpha=0.5)
            plt.legend()
            plt.title(col)
            plt.tight_layout()
        plt.savefig(f"../{DT_output}/DT_features.pdf")
    except Exception as e:
        print(col,e)

features = [
    "B_Tr_T_PIDK",
    "B_Tr_T_PIDP",
    "B_Tr_T_PIDe",
    "B_Tr_T_PIDmu",
    "B_Tr_T_BVIPSig",
            ]

start = time.time()

# Path to input root file
file_pattern = f'/eos/lhcb/user/m/miolocco/FT_NTuple/{config.sample_type}/2_added_features/*/notSelected.root'
file_paths = glob.glob(file_pattern)
print("Load Data: Start")

df = uproot.open(path).arrays(features+["B_Tr_T_absID", "B_Tr_T_Origin_Flag"],library = "pd" )
# Loop through each file and read the contents
for file_path in file_paths:
    with uproot.open(f'{file_path}:DecayTree').arrays(features+["B_Tr_T_absID", "B_Tr_T_Origin_Flag"],library = "pd" ) as file:
        # Loop through each key (object) in the file
        for key in file.keys():
            # Access the data associated with each key
            data = file[key].arrays()
            
            # Check if the key already exists in the dictionary
            if key in data_dict:
                # Concatenate the data if the key already exists
                data_dict[key] = uproot.concatenate([data_dict[key], data[key]])
            else:
                # Add the data to the dictionary if the key does not exist
                data_dict[key] = data[key] 

print(f"Load Data: Finished in {round(-start+ time.time() , 2)}s")

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
values = ["OSKaon", "OSMuon", "OSElectron", "SSPion", "SSProton", "SSKaon"]

df['particle_type'] = np.select(conditions, values)
print(f"Composition: \n {round(df.particle_type.value_counts()/df.shape[0],4)*100}")
df.loc[~df['particle_type'].isin(values), 'particle_type'] = 'not_taggingPart'

# Plot features
plot_features(df, values)
embed()

RIVEDERE
# Shuffle 
df = df.sample(frac=1)
x = df.loc[df.particle_type == 1][features + ["particle_type"]]
## Micol: added condition about IS_MUON cause Jonas additionally asked for this condition
#if tagger != 'Muon':
#    x = x[x['B_Tr_T_ISMUON']<=0.5]
#else:
#    x = x[x['B_Tr_T_ISMUON']>0.5]

# To get same amount of label 0 and label 1
x = pd.concat([x, df.loc[df.label == 0][features + ["particle_type"]].head(len(x))])
# Shuffle the data
y = x.particle_type

x.drop(columns="particle_type" , inplace = True)

x_train , x_test ,y_train, y_test= train_test_split(x, y, test_size = 0.3, random_state=42)

start = time.time()
print("Start fitting")
clf = tree.DecisionTreeClassifier(max_depth = 5)

clf.fit(x_train, y_train)
print(f"Fit in: {round(-start+ time.time() , 2)}s")

# If the Feature importance is needed
printFeatImport = True
if printFeatImport:
    feat_import = clf.tree_.compute_feature_importances(normalize=True)
    for i in range(len(feat_import)):
        print(features[i],round(100*feat_import[i],2))


#Check if directories exist
dir = 'plots'
dir_path = f'{config.repoPath}/{dir}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(f'{dir_path}/{eventType}'):
    os.makedirs(f'{dir_path}/{eventType}')
if not os.path.exists(f'{dir_path}/{eventType}/DT'):
    os.makedirs(f'{dir_path}/{eventType}/DT')
    
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
plt.savefig(f"{config.repoPath}plots/{eventType}/DT/{particle}_ROC_AUC.pdf")
#plt.show()
plt.close()


# Visualize the decision tree
dot_data = tree.export_graphviz(clf,feature_names=features,class_names=[f"Not{particle}",f"{particle}"],filled=True, rounded=True,special_characters=True  ) 
graph = graphviz.Source(dot_data) 
graph.render(f"{config.repoPath}plots/{eventType}/DT/{particle}_DT")

#print(f"Accuracy:{clf.score(x_test,y_test)}")

path_to_tuple = f'/eos/lhcb/user/m/miolocco/FT_NTuple/{config.sample_type}/2_added_features/*/notSelected.root:DecayTree;1'
for df in uproot.iterate(path_to_tuple, loading_variables, step_size=stepsize, library = 'pd'):
