import numpy as np
import torch
from torch.utils.data import DataLoader
import uproot
import pandas as pd
from inputDataset import inputDataset
import pickle
from IPython import embed
import os
import argparse
from pprint import pprint
import datetime
import yaml
import json
from os.path import join
from matplotlib import pyplot as plt
from scripts import ranges, nice_names, matplotlib_lhcb_style
# Local import
import scripts.pyTorchTraining as pyTrain
from scripts.NNModel import NeuralNetwork
matplotlib_lhcb_style(plt)
from scripts.preSelections import run2_taggers_variables
from scripts.adding_features_v2 import data_vars_translation

def plot_tagDec(tagger, df_TagParticles, plotPath):
    plt.figure()
    plt.yscale("log")
    print(f'Eta range is [{df_TagParticles[f"{tagger}_Eta"].min(), df_TagParticles[f"{tagger}_Eta"].max()}]')
    plt.hist(df_TagParticles.loc[(df_TagParticles[f"{tagger}_TagDec"] == -1)][f"{tagger}_Eta"] ,bins = 100 , density = True , histtype = "stepfilled" ,range=(df_TagParticles[f"{tagger}_Eta"].min(),df_TagParticles[f"{tagger}_Eta"].max()), color = "green" , alpha=0.5, label = f"Tag. dec: b")
    plt.hist(df_TagParticles.loc[(df_TagParticles[f"{tagger}_TagDec"] == 1)][f"{tagger}_Eta"] ,bins = 100 , density = True , histtype = "stepfilled" ,range=(df_TagParticles[f"{tagger}_Eta"].min(),df_TagParticles[f"{tagger}_Eta"].max()), color = "orange" , alpha=0.5, label = f"Tag. dec: anti-b")

    plt.grid()
    plt.xlabel(r"$\eta$",fontsize=24)
    plt.ylabel("Normalized number of tracks", fontsize=24)
    plt.legend(loc = "best", title = f'{len(df_TagParticles[(df_TagParticles[f"{tagger}_TagDec"] == -1)|(df_TagParticles[f"{tagger}_TagDec"] == 1)])} tagged events')
    plt.title(f"{tagger} mistag", fontsize=24)
    plot_name = f'{plotPath}/tagDec.pdf'
    print(f'Tagging decision plot saved at {plot_name}')
    plt.savefig(f"{plot_name}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add tagging decision and mistag',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--selected', help='Files with applied pre-selections', type=str)
    parser.add_argument('--config', help='', type=str, default='logit')
    parser.add_argument('--cut', help='Cut used', type=str)
    parser.add_argument('--link', help='Link fucntion used for calibration', type=str, default='logit', choices=('mistag','logit'))
    parser.add_argument('--taggedData', help='Name of data (tagged data)', type=str)
    parser.add_argument('--model', help='Path to where the NN models are saved up to cut type', type=str)
    parser.add_argument('--modelPrePath', help='Path to where the NN models are saved up to cut type', type=str)
    parser.add_argument('--decayType', help='Event decay for calibration', type=str)
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--features', help='Input features used for NN training',) 
    parser.add_argument('--arch', help='NN architecture', type=str, default='[64,64,64]')
    parser.add_argument('--seed', help='Seed for reproducibility', type=int, default=42)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--scaler', help='Path to the scaler', type=str)
    parser.add_argument('--transformer', help='Path to the transformer', type=str)
    
    # parser.add_argument('--jsonPath', help='Where the best tagger candidates configs are saved', type=str, default='/home/molocco/classical-taggers/best_tagger_candidates')
    parser.add_argument('--data_calib', action="store_true")
    parser.add_argument('--repo', help="Path to repository")

    cfg = parser.parse_args()
    pprint(cfg)

    _features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    features = []
    if cfg.data_calib:
        for v in _features:
            if "BPV" in v: v=v.replace("BPV", "OWNPV_").replace("OWNPV_IP", "OWNPVIP")
            if "END_V" in v: v=v.replace("END_V", "ENDV_")
            features.append(v) if v not in data_vars_translation.keys() else features.append(data_vars_translation[v])
    else:
        features = _features

    loading_variables = features+ run2_taggers_variables + ['entry','B_Tr_T_Charge','selected', 'RUNNUMBER', 'EVENTNUMBER']
    if cfg.data_calib: loading_variables += ["B_ID", "FillNumber", "B_DTF_PV_Jpsi_MASS", "B_DTF_PV_MASS"]
    else: loading_variables += ["B_TRUEID"]
    loading_variables = np.unique(loading_variables).tolist()
    print(f"The features used are: {features}")

    #Load the best model (ie with the lowest training loss) and evaluate it on the test set
    json_file=f'candidatedTaggers_{cfg.link}.json'
    #Read the best tagger candidate config from json file with the best hyperparameter combination
    trainOn = "Data" if cfg.data_calib else "MC"
    with open(f'{cfg.repo}/best_tagger_candidates/{cfg.cut}/{trainOn}/{json_file}', 'r') as f:
        data = json.load(f)
    seed = int(data[cfg.tagger]['seed'])
    lr = float(data[cfg.tagger]['learning_rate'])
    bs = int(data[cfg.tagger]['batch_size'])
    arch = data[cfg.tagger]['architecture']
    dm = float(data[cfg.tagger]['min_delta'])
    config = f'lr{lr}_bs{bs}_{arch}_dm{dm}'
    model_path = join(cfg.modelPrePath, f"{seed}/{config}")
    # Load YAML configuration file
    with open(f'{cfg.repo}/configs/{config}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    bestModel = NeuralNetwork(features=features, architecture=arch, seed=seed, optimizer_kwargs={"lr" : lr}, repo_path=cfg.repo)
    pyTrain.load_model(model=bestModel, target_path=model_path)
    bestModel.eval()

    ## To be removed
    #testSetPath = f"{cfg.model}/testSet.csv"
    #test_df = pd.read_csv(f"{testSetPath}")
    #test_df.rename(columns={'TagDec': f"{cfg.tagger}_TagDec"}, inplace=True)
    
    ## Data loading
    with uproot.open("{}".format(cfg.selected)) as f:
        test_df = f[cfg.treename].arrays(loading_variables, library="pd")    

    # Assignation of the tagging decision (d)
    # d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
    if ("Bd" or "Bs" in cfg.decayType) and (cfg.tagger == "SSKaon" or cfg.tagger == "SSPion" ):
        test_df[f"{cfg.tagger}_TagDec"] = test_df[f"B_Tr_T_Charge"]
    else:
        test_df[f"{cfg.tagger}_TagDec"] = test_df[f"B_Tr_T_Charge"] * (-1)
    # Assignation of the label (it will be used as NN output)
    # The label is given by the product of the tagging decision and the flavour charge of the B.
    # It indicates if the tagging decision is wrong or correct.
    # -1 == wrong tag  1 == correct tag
    # When using data:
    #   - tagging decision: the B_TRUEID must be replaced with B_ID 
    #   - calibration: B_ID = reconstructed ID when moving to data!

    # Now the label is needed for the scaling, but in the future must be removed before scaling in the training so that it'ds not necessary here 
    id_var = "B_TRUEID" if not cfg.data_calib else "B_ID"
    test_df["label"] = test_df[f"{cfg.tagger}_TagDec"] * test_df[id_var]/abs(test_df[id_var]) 
    test_df.loc[test_df.label == -1, "label"] = 0 # shifting the label from -1 to 0
    
    # Data pre-processing 
    scalerPath = f"{model_path}/st_scaler.pkl"
    transformerPath = f"{model_path}/powerTransformer.pkl"
    columns_to_drop = ['entry', id_var, 'B_Tr_T_Charge','selected', 'RUNNUMBER', 'EVENTNUMBER', f'{cfg.tagger}_TagDec']
    pyTrain.plot_features(data=test_df[test_df.selected==1], features_list=features, target_path= os.path.dirname(cfg.taggedData), flag='label', name=f'training_inputFeatures')

    #test_dataset = inputDataset(df=test_df.drop(columns = columns_to_drop))
    test_dataset = inputDataset(df=test_df[features+['label']])
    test_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

    print('Adding tagging decision')
    #test_df['Eta'] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df[f'{cfg.tagger}_Eta'] = 1 - bestModel.evaluate_model(test_dl)[0] # bestModel.evaluate_model returns predicted probabilities for label 1, true values

    # Assign tagging decision = 0 for tracks that don't pass the pre-selection
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_Eta"] = 0.5  # classic
    
    # Eta Normalization [0, 0.5]
    test_df.loc[test_df[f'{cfg.tagger}_Eta'] > 0.5 , f"{cfg.tagger}_TagDec"] *= -1
    test_df.loc[test_df[f'{cfg.tagger}_Eta'] > 0.5, f"{cfg.tagger}_Eta"] *= -1
    test_df.loc[test_df[f'{cfg.tagger}_Eta'] < 0, f"{cfg.tagger}_Eta"] += 1 

    # Take only tagging track with best mistag
    df_TagParticles = test_df.sort_values(by = ['selected',f'{cfg.tagger}_Eta'] , ascending = [False,True]).groupby(['entry', 'RUNNUMBER', 'EVENTNUMBER']).first().reset_index()
    plot_tagDec(tagger =cfg.tagger, df_TagParticles=df_TagParticles, plotPath=f'{os.path.dirname(cfg.taggedData)}')
    
    # Save the selected tracks into NTuples
    os.makedirs(os.path.dirname(cfg.taggedData), exist_ok=True)
    save_vars = ['entry', 'RUNNUMBER', 'EVENTNUMBER',  f'{cfg.tagger}_TagDec', f'{cfg.tagger}_Eta', id_var]+run2_taggers_variables
    if cfg.data_calib:
        save_vars += ["FillNumber", "B_DTF_PV_Jpsi_MASS", "B_DTF_PV_MASS"]
    with uproot.recreate(f"{cfg.taggedData}") as file:
        file["DecayTree"] = df_TagParticles[save_vars]
        #file["DecayTree"] = df_TagParticles[['event_entry', f'{cfg.tagger}_TagDec', f'{cfg.tagger}_Eta', 'B_TRUEID']]
        
    print(f'File created at {cfg.taggedData}')
    # To be done at the end! when reading all files!!
    #df_TagParticles = test_df.sort_values(by = ["selected", f"{cfg.tagger}_Eta"] , ascending = [False,True]).groupby("entry").first()
 