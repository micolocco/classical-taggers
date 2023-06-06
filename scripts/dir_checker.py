import numpy as np 
import os 
import sys

repoPath = np.genfromtxt(f"../config.txt", dtype = str, delimiter=",")[0]

# Check whether the specified directories exist, otherwise create them
directory_list = ["calibrationPlots", "plots", "results", "root"]
decays = ["Bs2DsPi", "Bd2JpsiKst", "Bu2JpsiK"]
taggers = ["OSKaon", "OSMuon", "OSElectron", "SSKaon", "SSPion", "SSProton"]
for dir in directory_list:
    dir_path = f"{repoPath}/{dir}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for eventType in decays:
        if not os.path.exists(f"{dir_path}/{eventType}"):
            os.makedirs(f"{dir_path}/{eventType}")
        for tagger in taggers:
            if not os.path.exists(f"{dir_path}/{eventType}/{tagger}"):
                os.makedirs(f"{dir_path}/{eventType}/{tagger}")
