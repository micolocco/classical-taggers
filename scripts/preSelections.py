import numpy as np
import pandas as pd
import uproot
import re


def extract_selection_var(cut_file):
    '''
    Function to extract strings from pre-selections cat. 
    Return vector of strings
    '''
    # Input string
    input_string = np.genfromtxt(f"{cut_file}.txt", dtype = str, delimiter=",")
    # Define the regular expression pattern
    pattern = r'\(([^<>=]+)[<>=]'
    # Use the findall function to extract all matches
    matches = re.findall(pattern, f"{input_string}")
    # Create an array with the extracted strings
    result_array = [match.strip() for match in matches]
    result_array = np.unique(result_array).tolist()
    return result_array

def apply_preSelections(notSelected_rootPath, cut_file, loading_variables, selected_rootPath):

    print(f"Applying pre-selections on sample: {notSelected_rootPath}")
    df = uproot.open(notSelected_rootPath).arrays(loading_variables,library = "pd" ) # load all data
    cuts = np.genfromtxt(f"{cut_file}.txt", dtype = str, delimiter=",")
    print(f"The applied cut is: {cuts}")
    df.eval(f"selected_track = {cuts}", inplace = True)
    df.selected_track = df.selected_track.astype(int, copy = False) 

    # Save the selected tracks into NTuples
    with uproot.recreate(f"{selected_rootPath}") as file:
        file["DecayTree"] = df
    print(f"Pre-selections applied. NTuple saved at {selected_rootPath}")
    return df