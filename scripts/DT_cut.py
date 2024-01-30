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

def perc_composition(df):
    df.eval(f"ID = abs(B_Tr_T_TRUEID)", inplace=True)
    print(f"List of different IDs: \n {df.ID.value_counts()}")
    print(f"Kaon: {round(df.ID[df.ID==321].count()/df.shape[0],2)*100}%")
    print(f"Muon: {round(df.ID[df.ID==13].count()/df.shape[0],2)*100}%")
    print(f"Electron: {round(df.ID[df.ID==11].count()/df.shape[0],2)*100}%")
    print(f"Pion: {round(df.ID[df.ID==211].count()/df.shape[0],2)*100}%")
    print(f"Proton: {round(df.ID[df.ID==2212].count()/df.shape[0],2)*100}%")

def plot_features(data, eventType, particle, folder='plots', nbins=100):

    # Plot input features 
    plt.figure(figsize=(24,50))
    try:
        for i, col in enumerate(data.columns.to_list()):
            plt.subplot(10, 3, i + 1)
            plt.hist(data[col][data['label']==0], density = True, bins=nbins, label = f"Not {particle}",color='b', alpha=0.5)
            plt.hist(data[col][data['label']==1], density = True, bins=nbins, label = f"{particle}",color='r', alpha=0.5)
            plt.legend()
            plt.title(col)
            plt.tight_layout()
        plt.savefig(f"../{folder}/{eventType}/DT/{particle}_features.pdf")
    except Exception as e:
        print(col,e)


eventType = sys.argv[1]
particle = sys.argv[2] # here just tagging particle type 

if "Kaon" in particle:
    abs_ID = 321
elif "Muon" in particle:
    abs_ID = 13
elif "Electron" in particle:
    abs_ID = 11
elif "Pion" in particle:
    abs_ID = 211
elif "Proton" in particle:
    abs_ID = 2212


features = [#"B_Tr_T_CHI2DOF",
                  #"B_Tr_T_ISMUON",
                  #"B_Tr_T_P",
                  #"B_Tr_T_PT",
                  #"B_Tr_T_GHOSTPROB",
                  #"B_Tr_T_PhiDistance",
                  #"B_Tr_T_PROBNN_E",
                  #"B_Tr_T_PROBNN_K",
                  #"B_Tr_T_PROBNN_P",
                  #"B_Tr_T_PROBNN_PI",
                  #"B_Tr_T_absIP",
                  "B_Tr_T_PIDK",
                  "B_Tr_T_PIDP",
                  "B_Tr_T_PIDe",
                  "B_Tr_T_PIDmu",
                  ]

start = time.time()

# Path to input root file
prefix = f'all_{config.sample_type}'
path = f"{config.repoPath}root/{eventType}/{prefix}_notSelected.root:DecayTree"
print("Load Data: Start")

df = uproot.open(path).arrays(features+["B_Tr_T_TRUEID"],library = "pd" ) 
print(f"Load Data: Finished in {round(-start+ time.time() , 2)}s")

perc_composition(df)
# 1 = tagger type of particle (electron, pion, proton, muon, kaon) 
df.eval(f"label = (abs(B_Tr_T_TRUEID/ {abs_ID}))",inplace= True) 

df.loc[df.label!= 1 , "label"] = 0

# Plot features

plot_features(df, eventType, particle)
# Shuffle 
df = df.sample(frac=1)
x = df.loc[df.label == 1][features + ["label"]]
## Micol: added condition about IS_MUON cause Jonas additionally asked for this condition
#if tagger != 'Muon':
#    x = x[x['B_Tr_T_ISMUON']<=0.5]
#else:
#    x = x[x['B_Tr_T_ISMUON']>0.5]

# To get same amount of label 0 and label 1
x = pd.concat([x, df.loc[df.label == 0][features + ["label"]].head(len(x))])
# Shuffle the data
y = x.label

x.drop(columns="label" , inplace = True)

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