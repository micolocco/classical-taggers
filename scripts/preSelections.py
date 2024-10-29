import numpy as np
import uproot
import re
from scripts.adding_features_v2 import loading_variables
import pyTorchTraining as pyTrain
import argparse
import os


def extract_selection_var(cut_file):
    '''
    Function to extract strings from pre-selections cat. 
    Return vector of strings
    '''
    # Input string
    input_string = np.genfromtxt(f"{cut_file}", dtype = str, delimiter=",")
    # Define the regular expression pattern
    pattern = r'\(+([^<>!=]+)[<>!=]'
    # Use the findall function to extract all matches
    matches = re.findall(pattern, f"{input_string}")
    # Create an array with the extracted strings
    result_array = [match.strip() for match in matches]
    result_array = np.unique(result_array).tolist()
    return result_array

def apply_preSelections(notSelected_rootPath, cut_file, treename, loading_variables):
    print(f"Applying pre-selections on sample: {notSelected_rootPath}")
    with uproot.open("{}".format(notSelected_rootPath)) as f:
        df = f[treename].arrays(loading_variables, library="pd")
    cuts = np.genfromtxt(f"{cut_file}", dtype = str, delimiter=",")
    print(f"The applied cut is: {cuts}")
    df.eval(f"selected = {cuts}", inplace = True)
    df.selected = df.selected.astype(int, copy = False) 
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--added_features', help='Added features file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')
    parser.add_argument('--cut_file', help='File where the cut is stored', type=str)
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--features', help='Input features for NN training', default='union_PROBNN') 

    cfg = parser.parse_args()

    from pprint import pprint
    pprint(cfg)

    selection_variables = extract_selection_var(cfg.cut_file) + ['entry', 'RUNNUMBER', 'EVENTNUMBER', 'B_TRUEID', 'B_Tr_T_Charge','B_Tr_T_TRUEID']
    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features)
    loading_variables = features + selection_variables
    loading_variables = np.unique(loading_variables).tolist()
    '''
    if "SS" in cfg.tagger:
        particle = cfg.tagger.removeprefix("SS")
        loading_variables += ["B_Tr_T_DeltaQ_" + particle]
        if particle in ("Proton", "Pion"):
            loading_variables += ["B_Tr_T_PIDP"]
    '''
    df = apply_preSelections(cfg.added_features, cfg.cut_file, cfg.treename, loading_variables)[features + ['entry', 'RUNNUMBER', 'EVENTNUMBER', 'B_TRUEID', 'B_Tr_T_Charge','selected']]

    # Save the selected tracks into NTuples
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with uproot.recreate(f"{cfg.output}") as file:
        file["DecayTree"] = df
    print(f"Pre-selections applied. NTuple saved at {cfg.output}")
