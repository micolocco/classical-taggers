import os
import json
import pandas as pd
import torch
from os.path import join
from torch.utils.data import DataLoader
import scripts.pyTorchTraining as pyTrain
from scripts.inputDataset import inputDataset


pre_path = "/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/"
cut = 'allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN'

taggers_dict = {
    #'OSKaon': 'Bu2JpsiK',
    'OSElectron': 'Bu2JpsiK',
    'OSMuon': 'Bu2JpsiK',
    'SSPion': 'Bd2JpsiKst',
    'SSProton': 'Bd2JpsiKst',
    'SSKaon': 'Bs2DsPi'
}
#Load the best model (ie with the lowest training loss) and evaluate it on the test set
json_file=f'candidatedTaggers_logit.json'
#Read the best tagger candidate config from json file with the best hyperparameter combination
with open(f'/home/molocco/classical-taggers/best_tagger_candidates/allBKGCAT_notSamePV_noOSP_SSK_balanced/{json_file}', 'r') as f:
    data = json.load(f)


for tagger in taggers_dict.keys():
    seed = int(data[tagger]['seed'])
    lr = float(data[tagger]['learning_rate'])
    bs = int(data[tagger]['batch_size'])
    arch = data[tagger]['architecture']
    dm = float(data[tagger]['min_delta'])
    config = f'lr{lr}_bs{bs}_{arch}_dm{dm}'

    decay = taggers_dict[tagger]
    pre_path_full = join(pre_path, f"{decay}/{tagger}/{cut}")
    model_path = join(pre_path_full, f"{seed}/{config}")
    model= f"{model_path}/model.pth"

    testSetPath = f"{model_path}/testSet.csv"
    test_df = pd.read_csv(f"{testSetPath}")
    columns_to_drop = ['event_entry', 'selected', f"{tagger}_TagDec", 'B_TRUEID',]
    print(f"Loading model for {tagger} on {decay} decay: {model}")
     # Load the entire model (with preprocessing already inside)
    bestModel = torch.load(model)
    bestModel.eval()

    test_dataset = inputDataset(df=test_df.drop(columns = columns_to_drop))
    # test_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

   
    #test_df[f"{tagger}_Eta"] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df['predictedProb'] = bestModel.evaluate_model(test_dl)[0] # bestModel.evaluate_model returns predicted probabilities for label 1, true values
    test_df[f"{tagger}_Eta"] = 1 - test_df['predictedProb']

    test_df = test_df[['event_entry','selected', f"{tagger}_Eta", f"{tagger}_TagDec", 'label','B_TRUEID']]
    # Assign tagging decision = 0 for tracks that don't pass the pre-selection
    test_df.loc[test_df.selected == 0, f"{tagger}_TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, f"{tagger}_Eta"] = 0.5  # classic
    #Eta Normalization [0, 0.5]
    test_df.loc[test_df[f'{tagger}_Eta'] > 0.5 , f"{tagger}_TagDec"] *= -1
    test_df.loc[test_df[f'{tagger}_Eta'] > 0.5, f"{tagger}_Eta"] *= -1
    test_df.loc[test_df[f'{tagger}_Eta'] < 0, f"{tagger}_Eta"] += 1 

    # Take only tagging track with best mistag
    df_TagParticles = test_df.sort_values(by = ["selected",f"{tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first()
    print(f"{df_TagParticles.shape[0]} tracks used for calibrating")
    target_path = f"{model_path}/calibration_plots_splitting"
    os.makedirs(target_path, exist_ok=True)

    pyTrain.calibration(tagger=tagger, df_tag=df_TagParticles, eventType=decay, target_path=target_path)
    # Try both calibration functions
    pyTrain.calibration(tagger=tagger, df_tag=df_TagParticles, eventType=decay, target_path=target_path, calibration_option='logit')
