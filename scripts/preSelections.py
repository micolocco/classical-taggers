import numpy as np
import pandas as pd
import uproot
import configParameters as config

def apply_preSelections(notSelected_rootPath, loading_variables, selected_rootPath, name_formatter, repoPath, eventType, tagger):

    print(f"Applying pre-selections on sample: {notSelected_rootPath}")
    df = uproot.open(notSelected_rootPath).arrays(loading_variables,library = "pd" ) # load all data
    cuts = read_cuts(name_formatter)
    print(f"The applied cut is: {cuts}")
    df.eval(f"selected_track = {cuts}", inplace = True)
    df.selected_track = df.selected_track.astype(int, copy = False) 
   
    # Save the selected tracks into NTuples
    with uproot.recreate(f"{selected_rootPath}") as file:
        file["DecayTree"] = df
    print(f"Pre-selections applied. NTuple saved at {selected_rootPath}")
    return df

def read_cuts(name_formatter):
    
    folder = "cuts"
    saveName = name_formatter.assign_name(folder, 'cut')
    cuts = np.genfromtxt(f"{saveName}.txt", dtype = str, delimiter=",")
    return cuts