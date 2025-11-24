import os
import json
import pandas as pd
import torch
from os.path import join
import scripts.pyTorchTraining as pyTrain
import uproot
import glob
from IPython import embed
from matplotlib import pyplot as plt

from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
    

pre_path = "/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/"
features = 'union_PROBNN'
cut = f'allBKGCAT_notSamePV_noOSP_SSK_balanced/{features}'
repo = '/home/molocco/classical-taggers'
taggers_dict = {
    'OSKaon': 'Bu2JpsiK',
    'OSElectron': 'Bu2JpsiK',
    'OSMuon': 'Bu2JpsiK',
    'SSPion': 'Bd2JpsiKst',
    'SSProton': 'Bd2JpsiKst',
    'SSKaon': 'Bs2DsPi'
}


for tagger in taggers_dict.keys():
    decay = taggers_dict[tagger]
    pre_path_full = join(pre_path, f"{decay}/{tagger}/{cut}")
    #model_path = join(pre_path_full, f"{seed}/{config}/{asymm}")
    #model= f"{model_path}/model.pth"
    base_dir = pyTrain.get_anchor_dir(model_path=pre_path_full, anchor=features)
    train_path = base_dir / "trainSet.parquet"
    if decay == "Bu2JpsiK":
        ID = 521
    elif decay == "Bd2JpsiKst": 
        ID = 511
    elif decay == "Bs2DsPi":
        ID = 531
    features = pyTrain.get_features(tagger=tagger, yaml_file='union_PROBNN', repo_path=repo)

    train_df = pd.read_parquet(train_path, engine="pyarrow")
    

    
    #for i, f in enumerate(input_files):
    #df = pd.DataFrame(columns=vars)
    #    print(f"Reading input file: {f}")
    #    with uproot.open("{}".format(f)) as _f:
    #        _df = _f['DecayTree'].arrays(vars+ ['RUNNUMBER', 'EVENTNUMBER'], library="pd")
    #    _df.dropna(inplace = True)
    #    _df["SAMPLENUMBER"] = i
    #    _df["event_entry"] = _df["SAMPLENUMBER"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
    #    _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'SAMPLENUMBER'], inplace=True)
    #    df = pd.concat([df, _df], ignore_index = True)
    #df.sample(frac=1, random_state=45).reset_index(drop=True) # cfg.seed
    #df = df.query('selected==1')
    ## Assignation of the tagging decision (d)
    ## d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
#
    #if ("Bd" in decay or "Bs" in decay) and (tagger == "SSKaon" or tagger == "SSPion" ):
    #    df[f"{tagger}_TagDec"] = df[f"B_Tr_T_Charge"]
    #else:
    #    df[f"{tagger}_TagDec"] = df[f"B_Tr_T_Charge"] * (-1)
    ## Assignation of the label (it will be used as NN output)
    ## The label is given by the product of the tagging decision and the flavour charge of the B.
    ## It indicates if the tagging decision is wrong or correct.
    ## -1 == wrong tag  1 == correct tag
    ## When using data:
    ##   - tagging decision: the B_TRUEID must be replaced with B_ID 
    ##   - calibration: B_ID = reconstructed ID when moving to data!
    ## id_var = "B_TRUEID" if not cfg.data_calib else "B_ID"
    #id_var = "B_TRUEID" 
#
    #df["label"] = df[f"{tagger}_TagDec"] * df[id_var]/abs(df[id_var])     
    #df.loc[df.label == -1, "label"] = 0 # shifting the label from -1 to 0
    
    
    #train_df, val_df, test_df = pyTrain.splitByEvent(df=df[features + ['event_entry', 'selected', f"{tagger}_TagDec", 'B_TRUEID', 'label']], asym_level=asymm,seed=3, train_val_split=0.9)

    target_path=f'{repo}/inputFeatures/{decay}/{tagger}/'
    os.makedirs(target_path, exist_ok=True)
    pyTrain.plot_features(data=train_df, features_list=features,  target_path=target_path, flag='label', name=f'inputFeatures')
    pyTrain.plot_features_byID(data=train_df, features_list=features, ID=ID, target_path=target_path, name=f'byTRUEID_inputFeatures')
    print(f"Plotted input features and saved to {target_path}")