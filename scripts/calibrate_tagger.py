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
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calibrate the tagger on the specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_path', help='Name of the output dir', type=str, default='../test')
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    #parser.add_argument('--seed', help='Random seed', default=2) 
    #parser.add_argument('--config', help='Config yaml', type=str, default='configs/config_test') 
    parser.add_argument('--decayType', help='Event decay', type=str)
    parser.add_argument('--repo', help="Path to repository", default='/home/molocco/classical-taggers', type=str)
    parser.add_argument('--features', help="Feature set", type=str, default='union_PROBNN')
    parser.add_argument('--simulation', help='If data are MC or real-data. Used for the calibration', action='store_true')
    parser.add_argument('--npar', help='Number of parameters for the calibration, default 2)', type=int, default=2)
    parser.add_argument('--function', help='Calibration function, either mistag or logit', type=str, choices=('mistag', 'logit'),default='mistag')
    print(f'Calibration started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg) 
    tagger = cfg.tagger
    if not cfg.simulation:
        if "Bu" in cfg.decayType:
            mode = "Bu"
        elif "Bd" in cfg.decayType:
            mode = "Bd"
        elif "Bs" in cfg.decayType:
            mode = "Bs"
    else:
        mode = "Bu" 

    if cfg.decayType == "Bu2JpsiK":
        ID = 521
    elif cfg.decayType == "Bd2JpsiKst": 
        ID = 511
    elif cfg.decayType == "Bs2DsPi":
        ID = 531
    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    print(f"Features used for training: {features}")
    #Load the best model (ie with the lowest training loss) and evaluate it on the test set
    model_name = f"{cfg.model_path}/model.pth"
    print(f"Loading model {model_name}")
    # Load the entire model (with preprocessing already inside)
    model = torch.load(model_name, weights_only=False)
    model.eval()
    
    # Load dataset
    base_dir = pyTrain.get_anchor_dir(model_path=cfg.model_path, anchor=cfg.features)
    test_path = base_dir / "testSet_full.parquet"
    train_path = base_dir / "trainSet.parquet" # to be removed if correct
    val_path = base_dir / "valSet.parquet" # to be removed if correct
    test_df  = pd.read_parquet(test_path,  engine="pyarrow")
    train_df = pd.read_parquet(train_path, engine="pyarrow") # to be removed if correct
    val_df   = pd.read_parquet(val_path,   engine="pyarrow") # to be removed if correct
    pyTrain.stats_printout_train_calib(train_df=train_df, val_df=val_df, test_df=test_df, tagger=cfg.tagger, ID=ID)
    # Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
    #test_df_sel1 = test_df.query('selected==1').copy()
    #test_dataset_sel1 = inputDataset(df=test_df_sel1.drop(columns = columns_to_drop))
    #test_dl_sel1 = DataLoader(test_dataset_sel1, batch_size = 1024, shuffle=False)
    #print(f"Test set has {len(test_dl_sel1.dataset)} tracks selected as tagging particles")
    #print(f"Test set has {test_df[(test_df['selected']==1)&(test_df['label']==0)].shape[0]} wrong tagged tracks, {test_df[(test_df['selected']==1)&(test_df['label']==1)].shape[0]} correctly tagged tracks")
    #
    #test_df_sel1['yPred'], test_df_sel1['yTrue'] = model.evaluate_model(test_dl_sel1)
    #pyTrain.plot_mistag(tagger=cfg.tagger, df=test_df_sel1, target_path=cfg.target_path, type = 'Test')
    
    columns_to_drop = ['event_entry', 'selected', f"{cfg.tagger}_TagDec", 'B_TRUEID',]
    test_dataset = inputDataset(df=test_df.drop(columns = columns_to_drop))
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)
    #test_df['Eta'] = clf.predict_proba(model.evaluate_model(test_dl)[0])[:,0]
    test_df['predictedProb'] = model.evaluate_model(test_dl)[0] # model.evaluate_model returns predicted probabilities for label 1, true values
    test_df[f'{cfg.tagger}_Eta'] = 1 - test_df['predictedProb']
    test_df = test_df[['event_entry','selected', f"{cfg.tagger}_Eta", f"{cfg.tagger}_TagDec", 'label','B_TRUEID']]
    #print(test_df.loc[test_df.selected == 1][f"{cfg.tagger}_Eta"]) 
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_Eta"] = 0.5  # classic
    #pyTrain.plot_tagDec(tagger =cfg.tagger, df_TagParticles=test_df.sort_values(by = ["selected",f"{cfg.tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first(), plot_name=f'{cfg.target_path}/Not_Normalized_TagDec.pdf')pyTrain.plot_tagDec(tagger =cfg.tagger, df_TagParticles=test_df.sort_values(by = ["event_entry","selected","Eta"] , ascending = [True,False,True]).groupby("event_entry").first(), output_file='Not_Normalized_TagDec.pdf', target_path=cfg.target_path)
    # Eta Normalization [0, 0.5]# Eta Normalization [0, 0.5]
    # Eta Normalization [0, 0.5]
    test_df.loc[test_df[f"{cfg.tagger}_Eta"] > 0.5 ,f"{cfg.tagger}_TagDec"] *= -1
    test_df.loc[test_df[f"{cfg.tagger}_Eta"] > 0.5, f"{cfg.tagger}_Eta"] *= -1
    test_df.loc[test_df[f"{cfg.tagger}_Eta"] < 0, f"{cfg.tagger}_Eta"] += 1
    df_TagParticles = test_df.sort_values(by = ["selected",f"{cfg.tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first()
    print(f"{df_TagParticles.shape[0]} tracks used for calibrating")
    # Calibrating the tagger and saving parameters
    target_path=f'{cfg.model_path}/calibration_npar{cfg.npar}'
    os.makedirs(target_path, exist_ok=True)
    pyTrain.calibration(tagger=cfg.tagger, df_tag=df_TagParticles, npar=cfg.npar, calibration_option=cfg.function, enlarge_scale=False, target_path=target_path, mode=mode)
    # Try both calibration functions
    #pyTrain.calibration(tagger=cfg.tagger, df_tag=df_TagParticles, eventType=cfg.decayType, target_path=cfg.target_path, calibration_option='logit')
    print(f'Calibration finished on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
