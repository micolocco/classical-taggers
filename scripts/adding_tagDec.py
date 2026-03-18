import numpy as np
from torch.utils.data import DataLoader
import uproot
import pandas as pd
from inputDataset import inputDataset
from IPython import embed
import os
import argparse
from pprint import pprint
import yaml
from os.path import join
from matplotlib import pyplot as plt
from scripts import ranges, nice_names, matplotlib_lhcb_style
# Local import
import scripts.pyTorchTraining as pyTrain
from scripts.NNModel import NeuralNetwork
matplotlib_lhcb_style(plt)
from scripts.preSelections import run2_taggers_variables
from scripts.train_tagger import get_architecture
from scripts.adding_features import translate_mc_names_to_data

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
    parser.add_argument('--decayType', help='Event decay for calibration', type=str)
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--features', help='Input features used for NN training',) 
    parser.add_argument('--seed', help='Seed for reproducibility', type=int, default=42)
    parser.add_argument('--scaler', help='Path to the scaler', type=str)
    parser.add_argument('--transformer', help='Path to the transformer', type=str)
    parser.add_argument('--data_type', help='Type of data: Data or MC', type=str, choices=('Data', 'MC'))
    parser.add_argument('--repo', help="Path to repository")
    parser.add_argument('--domain_adapted', help='If the model is domain adapted', action='store_true') 

    cfg = parser.parse_args()
    pprint(cfg)

    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    features = translate_mc_names_to_data(features, cfg.decayType, drop_mc_cols = False)

    loading_variables = features+ run2_taggers_variables + ['entry','B_Tr_T_Charge','selected', 
                                                            'RUNNUMBER', 'EVENTNUMBER', 'file_id', 
                                                            'label', f'{cfg.tagger}_TagDec', 'B_ID',
                                                            'B_DTF_PV_Jpsi_MASS']
    if cfg.data_type == 'Data':
        loading_variables += ["FillNumber", 'signal_weights']
        if 'Bu' not in cfg.decayType:
            loading_variables.append("B_DTF_PV_Jpsi_TAU")
        
    loading_variables = np.unique(loading_variables).tolist()
    print(f"The features used are: {features}")


    # Load YAML configuration file
    with open(cfg.config, 'r') as file:
        config = yaml.safe_load(file)

    bestModel = NeuralNetwork(features=features, architecture=get_architecture(config), seed=cfg.seed, optimizer_kwargs={"lr" : config['learning_rate']}, repo_path=cfg.repo)

    if not cfg.domain_adapted:
        pyTrain.load_model(model=bestModel, target_path=os.path.dirname(cfg.model))
    else:
        pyTrain.load_model_without_domain_classifier(model=bestModel, target_path=os.path.dirname(cfg.model))

    bestModel.eval()

    ## Data loading
    with uproot.open("{}".format(cfg.selected)) as f:
        test_df = f[cfg.treename].arrays(loading_variables, library="pd")    

    test_df['event_entry'] = test_df['file_id'].astype(str) + "_" + test_df['RUNNUMBER'].astype(str) + "_" + test_df['EVENTNUMBER'].astype(str)

    # Data pre-processing 
    scalerPath = cfg.scaler
    transformerPath = cfg.transformer


    columns_to_drop = ['entry', "B_ID", 'B_Tr_T_Charge','selected', 'RUNNUMBER', 'EVENTNUMBER', f'{cfg.tagger}_TagDec', 'file_id']

    test_dataset = inputDataset(df=test_df[features+['label']])
    test_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

    print('Adding tagging decision')
    test_df[f'{cfg.tagger}_Eta'] = 1 - bestModel.evaluate_model(test_dl)[0] # bestModel.evaluate_model returns predicted probabilities for label 1, true values

    # Assign tagging decision = 0 for tracks that don't pass the pre-selection
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_Eta"] = 0.5  # classic

    
    # Take only tagging track with best mistag
    test_df = test_df.sort_values(by = ['selected',f'{cfg.tagger}_Eta'] , ascending = [False,True]).groupby(['event_entry']).first().reset_index()
    print(f"Tagging efficiency: {len(test_df[test_df[f'{cfg.tagger}_TagDec'] != 0]) / len(test_df)}")
    
    # Save the selected tracks into NTuples
    os.makedirs(os.path.dirname(cfg.taggedData), exist_ok=True)
    save_vars = ['entry', 'RUNNUMBER', 'EVENTNUMBER','file_id',  f'{cfg.tagger}_TagDec', f'{cfg.tagger}_Eta', "B_ID"]+run2_taggers_variables
    if cfg.data_type == 'Data':
        save_vars += ["FillNumber", "B_DTF_PV_Jpsi_MASS", 'signal_weights']
        if 'Bu' not in cfg.decayType:
            save_vars.append("B_DTF_PV_Jpsi_TAU")
    with uproot.recreate(f"{cfg.taggedData}") as file:
        file["DecayTree"] = test_df[save_vars]
        
    print(f'File created at {cfg.taggedData}')
