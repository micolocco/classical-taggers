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

def apply_preSelections(notSelected_rootPath, cut_file, treename, loading_variables, BKG0, data_calib):
    print(f"Applying pre-selections on sample: {notSelected_rootPath}")
    with uproot.open("{}".format(notSelected_rootPath)) as f:
        df = f[treename].arrays(loading_variables, library="pd")
    cuts = np.genfromtxt(f"{cut_file}", dtype = str, delimiter=",")
    if data_calib:
        cuts = np.char.replace(cuts, "(B_Tr_T_Origin_Flag!=0)", "(B_Tr_T_IsInTree!=1)")
        cuts = np.char.replace(cuts, "BPV", "OWNPV")
        cuts = np.char.replace(cuts, "OWNPV_IP", "OWNPVIP")
    print(f"The applied cut is: {cuts}")
    df.eval(f"selected = {cuts}", inplace = True)
    if BKG0:
        print("Tracks with BKGCAT!=0 are removed")
        df = df[df.B_BKGCAT==0]
    df.selected = df.selected.astype(int, copy = False) 
    return df

run2_taggers_variables = [
        'B_Run2_SSPion_Dec',
        'B_Run2_SSPion_Omega',
        'B_Run2_SSPion_MVA',
        'B_Run2_SSKaon_Dec',
        'B_Run2_SSKaon_Omega',
        'B_Run2_SSKaon_MVA',
        'B_Run2_SSProton_Dec',
        'B_Run2_SSProton_Omega',
        'B_Run2_SSProton_MVA',
        'B_Run2_OSKaon_Dec',
        'B_Run2_OSKaon_Omega',
        'B_Run2_OSKaon_MVA',
        'B_Run2_OSElectron_Dec',
        'B_Run2_OSElectron_Omega',
        'B_Run2_OSElectron_MVA',
        'B_Run2_OSMuon_Dec',
        'B_Run2_OSMuon_Omega',
        'B_Run2_OSMuon_MVA',
    ]

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
    parser.add_argument('--BKG0', help='If specified, only BGKCAT=0 tracks are used',  action='store_true') # action='store_true' means args.BKG0 will be set to True if the --BKG0 argument is provided on the command line.
    parser.add_argument('--data_calib', action='store_true') # action='store_true' means args.BKG0 will be set to True if the --BKG0 argument is provided on the command line.
    parser.add_argument('--repo', help="Path to repository")
    #parser.add_argument('--run2_taggers', help='If specified, run2 taggers info is added',  action='store_true') # action='store_true' means args.BKG0 will be set to True if the --BKG0 argument is provided on the command line.

    cfg = parser.parse_args()

    from pprint import pprint
    pprint(cfg)

    selection_variables = extract_selection_var(cfg.cut_file)
    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    extra_variables = ['entry', 'RUNNUMBER', 'EVENTNUMBER', 'B_TRUEID', 'B_Tr_T_Charge']
    
    loading_variables = features + selection_variables + extra_variables + run2_taggers_variables
    loading_variables = np.unique(loading_variables).tolist()
    if cfg.BKG0: loading_variables += ['B_BKGCAT']

    if cfg.data_calib:
        loading_variables = [v for v in loading_variables if "TRUE" not in v and "Flag" not in v]
        loading_variables = [v.replace("BPV", "OWNPV_").replace("OWNPV_IP", "OWNPVIP") for v in loading_variables]
        loading_variables = [v.replace("END_V", "ENDV_") for v in loading_variables]
        loading_variables += ["B_Tr_T_IsInTree", "B_ID", "FillNumber", "B_DTF_PV_Jpsi_MASS", "B_DTF_PV_MASS"]

    # df = apply_preSelections(cfg.added_features, cfg.cut_file, cfg.treename, loading_variables, cfg.BKG0, cfg.data_calib)[features + extra_variables + run2_taggers_variables + ['selected']]
    df = apply_preSelections(cfg.added_features, cfg.cut_file, cfg.treename, loading_variables, cfg.BKG0, cfg.data_calib)

    # Save the selected tracks into NTuples
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with uproot.recreate(f"{cfg.output}") as file:
        file["DecayTree"] = df
    print(f"Pre-selections applied. NTuple saved at {cfg.output}")
