import numpy as np
import uproot
import re
import scripts.pyTorchTraining as pyTrain
import argparse
import os
from pprint import pprint
from adding_features import get_mass_label


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
    
def apply_preSelections(notSelected_rootPath, cut_file, treename, loading_variables, BKG0, data_type):
    print(f"Applying pre-selections on sample: {notSelected_rootPath}")
    with uproot.open("{}".format(notSelected_rootPath)) as f:
        df = f[treename].arrays(loading_variables, library="pd")
    cuts = np.genfromtxt(f"{cut_file}", dtype = str, delimiter=",")

    print(cuts)
    if cuts.size > 1:
        print('Parsing Cuts')
        cuts = ''.join(cuts)
        cuts = np.char.replace(cuts, "OR", "|")

    if data_type == 'Data':
        print("replacing B_Tr_T_Origin_Flag with B_Tr_T_IsInTree in the cut string for data")
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
        'B_Run2_SSKaon_Dec',
        'B_Run2_SSKaon_Omega',
        'B_Run2_SSProton_Dec',
        'B_Run2_SSProton_Omega',
        'B_Run2_OSKaon_Dec',
        'B_Run2_OSKaon_Omega',
        'B_Run2_OSElectron_Dec',
        'B_Run2_OSElectron_Omega',
        'B_Run2_OSMuon_Dec',
        'B_Run2_OSMuon_Omega',
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--to_select', help='Added features file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--cut_file', help='File where the cut is stored', type=str)
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--features', help='Input features for NN training', default='union_PROBNN') 
    parser.add_argument('--BKG0', help='If specified, only BGKCAT=0 tracks are used',  action='store_true') # action='store_true' means args.BKG0 will be set to True if the --BKG0 argument is provided on the command line.
    parser.add_argument('--data_type', help="Type of Data used, MC, Data or domain_adapted when using domain adaptation",choices=('MC', 'Data', 'domain_adapted'))
    parser.add_argument('--evtType', help='Decay which is being used', type=str, choices=('Bs2DsPi', 'Bd2JpsiKst', 'Bu2JpsiK', 'Bd2DmPi', 'Bs2JpsiPhi'))
    parser.add_argument('--repo', help="Path to repository")

    cfg = parser.parse_args()
    pprint(cfg)


    selection_variables = extract_selection_var(cfg.cut_file)
    tagger_features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    extra_variables = ['candidate_index', 'file_id', 'RUNNUMBER', 'EVENTNUMBER', 'B_ID', 'B_Tr_T_Charge']
    
    loading_variables = tagger_features + selection_variables + extra_variables + run2_taggers_variables
    loading_variables = np.unique(loading_variables).tolist()
    if cfg.BKG0: loading_variables += ['B_BKGCAT']
    loading_variables.append(get_mass_label(cfg.evtType))

    if cfg.data_type == 'Data':
        loading_variables += ["B_Tr_T_IsInTree", "FillNumber"]
        loading_variables.remove('B_Tr_T_Origin_Flag') # only in MC, replace with B_Tr_T_IsInTree in data

    if cfg.data_type == 'Data':
        loading_variables.append('B_TAU')

    loading_variables = list(dict.fromkeys(loading_variables)) #removes all duplicates
    print(loading_variables)
    df = apply_preSelections(cfg.to_select, cfg.cut_file, cfg.treename, loading_variables, cfg.BKG0, cfg.data_type)


    # Assignation of the tagging decision (d)
    # d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
    if ("Bd" or "Bs" in cfg.evtType) and (cfg.tagger == "SSKaon" or cfg.tagger == "SSPion" ):
        df[f"{cfg.tagger}_TagDec"] = df[f"B_Tr_T_Charge"]
    else:
        df[f"{cfg.tagger}_TagDec"] = df[f"B_Tr_T_Charge"] * (-1)

    # Assignation of the label (it will be used as NN output)
    # The label is given by the product of the tagging decision and the flavour charge of the B.
    # It indicates if the tagging decision is wrong or correct.
    # -1 == wrong tag  1 == correct tag


    df["label"] = df[f"{cfg.tagger}_TagDec"] * df['B_ID']/abs(df['B_ID']) 

    df.loc[df.label == -1, "label"] = 0 # shifting the label from -1 to 0

    #For domain adaptation data needs to unlabelled
    if cfg.data_type == 'domain_adapted':
        df.loc[df["domain"] == 0, "label"] = np.nan


    print(df.columns.tolist())
    # Save the selected tracks into NTuples
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with uproot.recreate(f"{cfg.output}") as file:
        file["DecayTree"] = df
    print(f"Pre-selections applied. NTuple saved at {cfg.output}")
