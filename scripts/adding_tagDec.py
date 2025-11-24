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
    parser.add_argument('--best_json', help='Json file with the best tagger candidates', type=str)
    parser.add_argument('--cut', help='Cut used', type=str)
    parser.add_argument('--link', help='Link function used for calibration', type=str, default='logit', choices=('mistag','logit'))
    parser.add_argument('--taggedData', help='Name of data (tagged data)', type=str)
    parser.add_argument('--modelPrePath', help='Path to where the NN models are saved up to cut type', type=str)
    parser.add_argument('--decayType', help='Event decay for calibration', type=str)
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--features', help='Input features used for NN training',) 
    parser.add_argument('--data_calib', action="store_true")
    parser.add_argument('--repo', help="Path to repository")
    parser.add_argument('--asymmetry_level', help='Asymmetry between B and Bbar with wrong and correct label. asym_level2: asymmetry in training and calibration samples, asym_level1 only calibration, asym_level0 none', default='asym_level1', type=str) 
    parser.add_argument('--signal_weights', action='store_true', help='store signal_weights if they are already in the NTuples') # action='store_true' means args.signal_weights will be set to True if the --signal_weights argument is provided on the command line.

    cfg = parser.parse_args()
    pprint(cfg)
    print(f'Adding tagging decision started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    loading_variables = features+ run2_taggers_variables + ['entry','B_Tr_T_Charge','selected', 'RUNNUMBER', 'EVENTNUMBER']
    #loading_variables = features + ['entry','B_Tr_T_Charge','selected', 'RUNNUMBER', 'EVENTNUMBER']
    if cfg.data_calib: 
        loading_variables += ["B_ID", "FillNumber", "B_DTF_PV_MASS", "B_DTF_PV_CTAU"]
        if cfg.signal_weights: 
            loading_variables += ["signal_weights"]
        if cfg.decayType:
            if "Jpsi" in cfg.decayType:
                loading_variables.append("B_DTF_PV_Jpsi_MASS")
                #loading_variables.append("B_DTF_PV_Jpsi_MASSERR")
            elif "Ds" in cfg.decayType:
                loading_variables.append("B_DTF_PV_Ds_MASS") 
                #loading_variables.append("B_DTF_PV_Ds_MASSERR")

    else: 
        loading_variables += ["B_TRUEID"]
    loading_variables = np.unique(loading_variables).tolist()
    print(f"The features used are: {features}")

   
    #Read the best tagger candidate config from json file with the best hyperparameter combination
    if cfg.best_json:
        json_file = cfg.best_json
    else:
        if cfg.data_calib:
            # Read models trained on data for calibration
            json_file = f'{cfg.repo}/best_tagger_candidates/{cfg.cut}/{cfg.features}/{cfg.asymmetry_level}/full/candidatedTaggers_overall_large_nominal.json'
        else:
            # Read models witj hold out sample for combination on MC (models are trained on MC)
            json_file = f'{cfg.repo}/best_tagger_candidates/{cfg.cut}/{cfg.features}/{cfg.asymmetry_level}/hold_out/candidatedTaggers_overall_large_nominal.json'

    with open(json_file, 'r') as f:
        data = json.load(f)
    seed = int(data[cfg.tagger]['seed'])
    lr = float(data[cfg.tagger]['learning_rate'])
    bs = int(data[cfg.tagger]['batch_size'])
    nL = data[cfg.tagger]['numlayers']
    nN = int(data[cfg.tagger]['numneurons'])
    config = f'lr{lr}_bs{bs}_nL{nL}_nN{nN}'
    model_path = join(cfg.modelPrePath, f"{seed}/{config}/{cfg.asymmetry_level}/hold_out") # asymmetry level hard coded for now, to be changed in the future
    #model_path = join(cfg.modelPrePath)
    # Load YAML configuration file
    with open(f'{cfg.repo}/configs/{config}.yaml', 'r') as file:
        print(file)
        config = yaml.safe_load(file)
    # bestModel = NeuralNetwork(features=features, numlayers=nL, seed=seed, optimizer_kwargs={"lr" : lr}, repo_path=cfg.repo)
    # bestModel = NeuralNetwork(features=features, numlayers=config['numlayers'], preprocess=preprocess_module, seed=cfg.seed, optimizer_kwargs={"lr" : config['learning_rate']}, repo_path=cfg.repo)
    # pyTrain.load_model(model=bestModel, target_path=model_path)
    # bestModel.eval()
    model_path = f"{model_path}/model.pth"

    # Load the entire model (with preprocessing already inside)
    bestModel = torch.load(model_path, weights_only=False)
    bestModel.eval()

    ## To be removed
    #testSetPath = f"{model_path}/testSet.csv"
    #test_df = pd.read_csv(f"{testSetPath}")
    #test_df.rename(columns={'TagDec': f"{cfg.tagger}_TagDec"}, inplace=True)
    
    ## Data loading
    with uproot.open("{}".format(cfg.selected)) as f:
        test_df = f[cfg.treename].arrays(loading_variables, library="pd")    

    # Assignation of the tagging decision (d)
    # d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
    if ("Bd" in cfg.decayType or "Bs" in cfg.decayType) and (cfg.tagger == "SSKaon" or cfg.tagger == "SSPion" ):
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
    # id_var = "B_TRUEID" if not cfg.data_calib else "B_ID"
    id_var = "B_TRUEID" if not cfg.data_calib else "B_ID"
    test_df["label"] = test_df[f"{cfg.tagger}_TagDec"] * test_df[id_var]/abs(test_df[id_var])     
    test_df.loc[test_df.label == -1, "label"] = 0 # shifting the label from -1 to 0
    
    # Data pre-processing 
    # scalerPath = f"{model_path}/st_scaler.pkl"
    # transformerPath = f"{model_path}/powerTransformer.pkl"
    # columns_to_drop = ['entry', id_var,'B_Tr_T_Charge','selected', 'RUNNUMBER', 'EVENTNUMBER', f'{cfg.tagger}_TagDec']
    columns_to_drop = ['entry', id_var,'B_Tr_T_Charge', 'RUNNUMBER', 'EVENTNUMBER', f'{cfg.tagger}_TagDec']

    #test_dataset = inputDataset(df=test_df.drop(columns = columns_to_drop))
    test_dataset = inputDataset(df=test_df[features+['label']])
    # test_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

    print('Adding tagging decision')
    #test_df['Eta'] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df[f'{cfg.tagger}_Eta'] = 1 - bestModel.evaluate_model(test_dl)[0] # bestModel.evaluate_model returns predicted probabilities for label 1, true values

    # Assign tagging decision = 0 for tracks that don't pass the pre-selection
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_Eta"] = 0.5  # classic
    #Eta Normalization [0, 0.5]
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
        save_vars += ["FillNumber", "B_DTF_PV_MASS", "B_DTF_PV_CTAU"]
        if "Jpsi" in cfg.decayType:
            save_vars.append("B_DTF_PV_Jpsi_MASS")
            #loading_variables.append("B_DTF_PV_Jpsi_MASSERR")
        elif "Ds" in cfg.decayType:
            save_vars.append("B_DTF_PV_Ds_MASS") 
            #loading_variables.append("B_DTF_PV_Ds_MASSERR")

        if cfg.signal_weights: 
            save_vars += ["signal_weights", "reweighter_weights"]
    with uproot.recreate(f"{cfg.taggedData}") as file:
        file["DecayTree"] = df_TagParticles[save_vars]
        #file["DecayTree"] = df_TagParticles[['event_entry', f'{cfg.tagger}_TagDec', f'{cfg.tagger}_Eta', 'B_TRUEID']]
        
    print(f'File created at {cfg.taggedData}')
    # To be done at the end! when reading all files!!
    #df_TagParticles = test_df.sort_values(by = ["selected", f"{cfg.tagger}_Eta"] , ascending = [False,True]).groupby("entry").first()
 