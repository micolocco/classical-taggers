import numpy as np
import uproot
import re
from scripts.adding_features_v2 import loading_variables
import scripts.pyTorchTraining as pyTrain
import argparse
import os

from IPython import embed
import re
import numpy as np

def extract_selection_var(cut_file):
    '''
    Extract variable names from a pre-selection file with OR-ed cut lines.
    Each line is a separate AND clause.
    '''
    with open(cut_file, "r") as f:
        cut_lines = [line.strip() for line in f if line.strip()]

    # regex pattern: match variable name before a comparator (e.g., <=, >=, <, >, ==, !=)
    pattern = r'\(*([\w\d_]+)\s*(?:<=|>=|<|>|==|!=)'

    all_vars = []
    for line in cut_lines:
        matches = re.findall(pattern, line)
        all_vars.extend([m.strip() for m in matches])

    return sorted(set(all_vars))  # unique, sorted list



def apply_preSelections(notSelected_rootPath, cut_file, treename, loading_variables, BKG0, data_calib):
    print(f"Applying pre-selections on sample: {notSelected_rootPath}")
    
    # Load variables from ROOT file
    with uproot.open(notSelected_rootPath) as f:
        df = f[treename].arrays(loading_variables, library="pd")

    # Read cuts line-by-line and build OR combination
    with open(cut_file, "r") as f:
        cut_lines = [line.strip() for line in f if line.strip() and line.strip().upper() != "OR"]

    print("Parsed cut lines:")
    for cut in cut_lines:
        print(f"  - {cut}")

    combined_cut = " | ".join(f"({cut})" for cut in cut_lines)
    if data_calib:
        combined_cut = combined_cut.replace("(B_Tr_T_Origin_Flag !=0)", "(B_Tr_T_IsInTree != 1)") # This is a workaround for the data calibration. The cut is not applied to the MC, but it is applied to the data. Actually, it's already applied in added_features_v2.py so there shouldn't be any InTree=1 in the data.
    print(f"Evaluating combined cut: {combined_cut}")
    df.eval(f"selected = {combined_cut}", inplace=True)

    # Filter by BKGCAT==0 if needed
    if BKG0:
        if 'Bs2DsPi' in notSelected_rootPath:
            print("Filtering out B_BKGCAT != 20") #They are the 95% of the events
            df = df[df.B_BKGCAT == 20]
        else:  
            print("Filtering out B_BKGCAT != 0")
            df = df[df.B_BKGCAT == 0]

    df.selected = df.selected.astype(int, copy=False)
    return df

run2_taggers_variables = [
        'B_Run2_SSPion_Dec',
        'B_Run2_SSPion_Omega',
        #'B_Run2_SSPion_MVA',
        'B_Run2_SSKaon_Dec',
        'B_Run2_SSKaon_Omega',
        #'B_Run2_SSKaon_MVA',
        'B_Run2_SSProton_Dec',
        'B_Run2_SSProton_Omega',
        #'B_Run2_SSProton_MVA',
        'B_Run2_OSKaon_Dec',
        'B_Run2_OSKaon_Omega',
        #'B_Run2_OSKaon_MVA',
        'B_Run2_OSElectron_Dec',
        'B_Run2_OSElectron_Omega',
        #'B_Run2_OSElectron_MVA',
        'B_Run2_OSMuon_Dec',
        'B_Run2_OSMuon_Omega',
        #'B_Run2_OSMuon_MVA',
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
    parser.add_argument('--data_calib', action='store_true') # action='store_true' means args.data_calib will be set to True 
    parser.add_argument('--repo', help="Path to repository")
    parser.add_argument('--signal_weights', action='store_true', help='store signal_weights if they are already in the NTuples') # action='store_true' means args.signal_weights will be set to True if the --signal_weights argument is provided on the command line.
    #parser.add_argument('--run2_taggers', help='If specified, run2 taggers info is added',  action='store_true') # action='store_true' means args.BKG0 will be set to True if the --BKG0 argument is provided on the command line.

    cfg = parser.parse_args()

    from pprint import pprint
    pprint(cfg)

    selection_variables = extract_selection_var(cfg.cut_file)
    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    extra_variables = ['entry', 'RUNNUMBER', 'EVENTNUMBER', 'B_TRUEID', 'B_Tr_T_Charge',]
    
    if cfg.BKG0: extra_variables += ['B_BKGCAT']
    if cfg.data_calib: extra_variables += ["B_Tr_T_IsInTree", "B_ID", "FillNumber", "B_DTF_PV_Jpsi_MASS", "B_DTF_PV_MASS", "B_DTF_PV_CTAU"]
    if cfg.signal_weights: extra_variables += ["signal_weights"]

    loading_variables = features + selection_variables + extra_variables + run2_taggers_variables
    # Remove eventual MC info
    if cfg.data_calib: loading_variables = [v for v in loading_variables if "TRUE" not in v and "Flag" not in v and "MC" not in v]
    
    loading_variables = np.unique(loading_variables).tolist()
    print("Loading variables are: ", loading_variables)
    df = apply_preSelections(cfg.added_features, cfg.cut_file, cfg.treename, loading_variables, cfg.BKG0, cfg.data_calib)#[features + extra_variables + run2_taggers_variables + ['selected']]
    print(f"Selected {df.shape[0]} events after pre-selection")
    
    # Save the selected tracks into NTuples
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    print(df.columns)
    with uproot.recreate(f"{cfg.output}") as file:
        file["DecayTree"] = df
    print(f"Pre-selections applied. NTuple saved at {cfg.output}")
    
