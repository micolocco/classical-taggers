import numpy as np
import pandas as pd
import uproot

def apply_preSelections(loading_variables, selected_rootPath, name_formatter, repoPath, eventType, tagger):

    decayPath = f"{repoPath}root/{eventType}/notSelected.root:DecayTree"
    df = uproot.open(decayPath).arrays(loading_variables,library = "pd" ) # load all data
    cuts = read_cuts(name_formatter)
    df.eval(f"selected_track = {cuts}", inplace = True)
    df.selected_track = df.selected_track.astype(int, copy = False) 
   
    # Save the selected tracks into NTuples
    with uproot.recreate(f"{selected_rootPath}") as file:
        file["DecayTree"] = df
    print(f"Saved NTuple with pre-selections at {selected_rootPath}")
    return df

def read_cuts(name_formatter):
    
    folder = "cuts"
    saveName = name_formatter.assign_name(folder, 'cut')
    cuts = np.genfromtxt(f"{saveName}.txt", dtype = str, delimiter=",")
    return cuts