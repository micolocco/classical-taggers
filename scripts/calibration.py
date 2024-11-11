import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import pandas as pd
from inputDataset import inputDataset
import pickle
from matplotlib import pyplot as plt
from IPython import embed
import os
import argparse
from pprint import pprint
import datetime
import yaml
# Local import
import scripts.pyTorchTraining as pyTrain
from scripts.NNModel import NeuralNetwork
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calibrate the tagger on the specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='../test')
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    #parser.add_argument('--seed', help='Random seed', default=2) 
    parser.add_argument('--config', help='Config yaml', type=str, default='configs/config_test') 
    parser.add_argument('--decayType', help='Event decay', type=str)
    print(f'Calibration started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg) 
    tagger = cfg.tagger
    # Path to where the scaler parameters are saved
    scalerPath = f"{cfg.target_path}/st_scaler.pkl"
    transformerPath = f"{cfg.target_path}/powerTransformer.pkl"

    # Path to where the test set is saved
    testSetPath = f"{cfg.target_path}/testSet.csv"
    test_df = pd.read_csv(f"{testSetPath}")

    #Load the best model (ie with the lowest training loss) and evaluate it on the test set
    bestModel = NeuralNetwork(features=features, optimizer_kwargs={"lr" : config.learning_rate}).to(device)
    pyTrain.load_model(bestModel, cfg.target_path)
    bestModel.eval()
    
    # Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
    test_df_sel1 = test_df.query('selected==1').copy()
    test_dataset_sel1 = inputDataset(df=test_df_sel1.drop(columns = columns_to_drop))
    test_dataset_sel1.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl_sel1 = DataLoader(test_dataset_sel1, batch_size = 1024, shuffle=False)
    print(f"Test set has {len(test_dl_sel1.dataset)} tracks selected as tagging particles")
    print(f"Test set has {test_df[(test_df['selected']==1)&(test_df['label']==0)].shape[0]} wrong tagged tracks, {test_df[(test_df['selected']==1)&(test_df['label']==1)].shape[0]} correctly tagged tracks")
    
    test_df_sel1['yPred'], test_df_sel1['yTrue'] = bestModel.evaluate_model(test_dl_sel1)
    pyTrain.plot_ROC(tagger=cfg.tagger, val_df=test_df_sel1, target_path =cfg.target_path)
    plt.figure()
    plt.hist(1-test_df_sel1['yPred'],bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Test set: Probability of label 0, only selected")
    plt.savefig(f"{cfg.target_path}/testSet_prob0distrib.pdf")
    plt.figure()
    plt.hist(test_df_sel1['yPred'],bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Test set: Probability of label 1")
    plt.savefig(f"{cfg.target_path}/testSet_prob1distrib.pdf")
    pyTrain.plot_mistag(tagger=cfg.tagger, df=test_df_sel1, target_path=cfg.target_path, type = 'Test')
    

    test_dataset = inputDataset(df=test_df.drop(columns = columns_to_drop))
    test_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

    #test_df['Eta'] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df['predictedProb'] = bestModel.evaluate_model(test_dl)[0] # bestModel.evaluate_model returns predicted probabilities for label 1, true values
    test_df['Eta'] = 1 - test_df['predictedProb']

    test_df = test_df[['event_entry','selected', 'Eta', 'TagDec', 'label','B_TRUEID']]

    #print(test_df.loc[test_df.selected == 1].Eta) 

    test_df.loc[test_df.selected == 0, "TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, "Eta"] = 0.5  # classic
    pyTrain.plot_tagDec(tagger =cfg.tagger, df_TagParticles=test_df.sort_values(by = ["event_entry","selected","Eta"] , ascending = [True,False,True]).groupby("event_entry").first(), output_file='Not_Normalized_TagDec.pdf', target_path=cfg.target_path)

    # Eta Normalization [0, 0.5]
    test_df.loc[test_df.Eta > 0.5 ,"TagDec"] *= -1
    test_df.loc[test_df.Eta > 0.5, "Eta"] *= -1
    test_df.loc[test_df.Eta < 0, "Eta"] += 1

    df_TagParticles = test_df.sort_values(by = ["event_entry","selected","Eta"] , ascending = [True,False,True]).groupby("event_entry").first()

    print(f"{df_TagParticles.shape[0]} tracks used for calibrating")
    pyTrain.plot_tagDec(tagger =cfg.tagger, df_TagParticles=df_TagParticles, target_path=cfg.target_path)
    # Calibrating the tagger and saving parameters
    pyTrain.calibration(tagger=cfg.tagger, df_tag=df_TagParticles, eventType=cfg.decayType, target_path=cfg.target_path)
    # Try both calibration functions
    pyTrain.calibration(tagger=cfg.tagger, df_tag=df_TagParticles, eventType=cfg.decayType, target_path=cfg.target_path, calibration_option='logit')
    print(f'Calibration finished on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
