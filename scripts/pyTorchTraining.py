import numpy as np 
import sys 
import pandas as pd 
import os

# Decay and tagger type are given as inputs by the user 
eventType = sys.argv[1]
tagger = sys.argv[2]

# Definition of the features and the variables used for pre-selection cuts (outputs od the Decision Tree)
taggers = ["OSKaon", "OSMuon", "OSElectron", "SSKaon", "SSPion", "SSProton"]

if tagger in taggers:
    
    features = ["B_Tr_T_cos_diff_Phi",
            "B_Tr_T_PhiDistance",
            "B_Tr_T_PT",
            "B_Tr_T_CHI2DOF",
            "B_Tr_T_BPVIP",
            "B_Tr_T_GHOSTPROB",
            "B_Tr_T_BPVIPCHI2",
            "diff_P",
            "B_Tr_T_EtaDistance",
            "P_proj",
            "EVIP"]

    selection_variables = ["entry", "label" , "B_Tr_T_PROBNN_K", "B_Tr_T_P" , "B_Tr_T_ISMUON", "B_Tr_T_BPVX" , "B_BPVX" , 
                         "B_Tr_T_BPVY" ,"B_BPVY" ,"B_Tr_T_BPVZ" , "B_BPVZ","B_Tr_T_Charge", "B_TRUEID", "B_Tr_T_absIP",
                         "B_Tr_T_PROBNN_PI", "B_Tr_T_PROBNN_P", "B_Tr_T_PROBNN_E", "B_Tr_T_PIDe", "B_Tr_T_PIDK", "B_Tr_T_PIDmu"]

    features += selection_variables

    if "SS" in tagger:
        particle = tagger.removeprefix("SS")
        features += ["B_Tr_T_DeltaQ_"+particle]
        if particle in ("Proton", "Pion"):
            features += ["B_Tr_T_PIDP"]

# Check whether the specified directories exist, otherwise create them
directory_list = ['calibrationPlots', 'plots', 'results', 'root']

for dir in directory_list:
    dir_path = f'{repoPath}/{dir}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(f'{dir_path}/{eventType}'):
        os.makedirs(f'{dir_path}/{eventType}')
    if not os.path.exists(f'{dir_path}/{eventType}/{tagger}'):
        os.makedirs(f'{dir_path}/{eventType}/{tagger}')