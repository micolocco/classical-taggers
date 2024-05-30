import numpy as np
import uproot
import pandas as pd
import os
import argparse
from pprint import pprint

# Define a function to find the corresponding bin and pick a random PID
def get_bin_label(momentum, bin_edges, pid_dict, max, min):
    if momentum>max:
        mom=max
    elif momentum<min:
        mom=min
    else:
        mom=momentum
    bin_index = np.digitize([mom], bin_edges, right=True) - 1
    bin_index = bin_index[0]  # Extract the scalar from the array
    bin_label = [k for i, k in enumerate(pid_dict.keys()) if i == bin_index]
    b = bin_label[0]

# Function to get a random PID from the corresponding bin
def get_random_pid(momentum, bin_edges, pid_dict, max, min):
    bin_label = get_bin_label(momentum, bin_edges, pid_dict, max. min)
    return np.random.choice(pid_dict[bin_label])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--file', help='File to which add the DLLs')
    parser.add_argument('--treename', help='TreeName of the input file', type=str, default='Tuple/DecayTree')
    parser.add_argument('--torch_dlls', help='DLLs from TORCH studies', type=str, nargs="+")
    parser.add_argument('--branch_name', help='Name of the added TORCH DLLs branch', type=str, default="logSumProtonMinusKaon_TORCH")
    parser.add_argument('--output', help='File where the DLL will be stored', type=str)

    cfg = parser.parse_args()
    pprint(cfg)

    # Initialize the random seed
    np.random.seed(42)

    with uproot.open("{}".format(cfg.file)) as f:
        input_df = f[cfg.treename].arrays(["B_Tr_T_P"], library="pd")

    torch_df = pd.DataFrame(columns=["B_Tr_T_P", "logSumProton", "logSumKaon"])
    for f in cfg.torch_dlls:
        print(f"Reading TORCH DLLs file: {f}")
        with uproot.open("{}".format(f)) as _f:
            _torch_df= _f["PIDAlgorithm_particle_output"].arrays(["momentum", "logSumProton", "logSumKaon"], library="pd")
        torch_df = pd.concat([torch_df, _torch_df], ignore_index = True)

    torch_df["logSumProtonMinusKaon"] = torch_df["logSumProton"] - torch_df["logSumKaon"]
    torch_df["momentum"] = torch_df["momentum"]*1000 # GeV to MeV conversion
    bins, bin_edges = pd.qcut(torch_df['momentum'], q=10000, retbins=True, duplicates='drop')
    torch_df['momentum_bin'] = bins
    torch_grouped = torch_df.groupby('momentum_bin')['logSumProton'].apply(list).reset_index()
    torch_grouped = torch_df.groupby('momentum_bin')['logSumKaon'].apply(list).reset_index()
    torch_grouped = torch_df.groupby('momentum_bin')['logSumProtonMinusKaon'].apply(list).reset_index()
    max = np.max(torch_df['momentum'])
    min = np.min(torch_df['momentum'])

    pid_dict_minus = torch_grouped.set_index('momentum_bin')['logSumProtonMinusKaon'].to_dict()
    # Apply the function to assign a random PID to each row
    input_df['logSumProtonMinusKaon'] = input_df['B_Tr_T_P'].apply(lambda x: get_random_pid(x, bin_edges, pid_dict_minus, max, min))

    pid_dict_kaons = torch_grouped.set_index('momentum_bin')['logSumKaon'].to_dict()
    input_df['logSumKaon'] = input_df['B_Tr_T_P'].apply(lambda x: get_random_pid(x, bin_edges, pid_dict_kaons, max, min))

    pid_dict_protons = torch_grouped.set_index('momentum_bin')['logSumProton'].to_dict()
    input_df['logSumProton'] = input_df['B_Tr_T_P'].apply(lambda x: get_random_pid(x, bin_edges, pid_dict_protons, max, min))

    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with uproot.recreate(cfg.output) as f:
        f[cfg.treename] = input_df['logSumProtonMinusKaon', 'logSumKaon', 'logSumProton']