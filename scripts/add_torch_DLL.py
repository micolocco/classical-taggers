import numpy as np
import uproot
import pandas as pd
import os
import argparse
from pprint import pprint

# Define a function to find the corresponding bin from the momenta of the tagging track
def get_bin_label(momentum, bin_edges, pid_dict, max, min):
    if momentum>max:
        mom=max
    elif momentum<min:
        mom=min+0.001
    else:
        mom=momentum
    bin_index = np.digitize([mom], bin_edges, right=True) - 1
    bin_index = bin_index[0]  # Extract the scalar from the array
    bin_label = [k for i, k in enumerate(pid_dict.keys()) if i == bin_index]
    b = bin_label[0]
    return b

# Function to get a random PID of a track, for a given TRUEID and momenta
def get_random_pid(momentum, truepid, torch_particle_dic):
    possible_ids = [p['TRUEID'] for p in torch_particle_dic.values()]
    possible_ids+=[-p for p in possible_ids]
    # truepid = int(truepid) if truepid in possible_ids else 211
    truepid = int(truepid) if truepid in possible_ids else 321
    for particle in torch_particle_dic.keys():
        if truepid==torch_particle_dic[particle]['TRUEID'] or truepid==-torch_particle_dic[particle]['TRUEID']:
            bin_label = get_bin_label(
                momentum,
                torch_particle_dic[particle]['bin_edges'],
                torch_particle_dic[particle]['pid_dict_minus'],
                torch_particle_dic[particle]['max_mom'],
                torch_particle_dic[particle]['min_mom'],
                )
            return np.random.choice(torch_particle_dic[particle]['pid_dict_minus'][bin_label])


# Function to create equipopulated bins of the variable var, for each particle type
def create_bins(df, var, torch_particle_dic):
    pid_cut = 'truePID == {trueid} or truePID == -{trueid}'.format(trueid=torch_particle_dic['TRUEID'])
    df = df.query(pid_cut)
    bins, bin_edges = pd.qcut(df[var], q=torch_particle_dic['nbins'], retbins=True, duplicates='drop')
    df[f'{var}_bins'] = bins
    df_grouped = df.groupby(f'{var}_bins')['logSumProton'].apply(list).reset_index()
    df_grouped = df.groupby(f'{var}_bins')['logSumKaon'].apply(list).reset_index()
    df_grouped = df.groupby(f'{var}_bins')['logSumProtonMinusKaon'].apply(list).reset_index()
    torch_particle_dic['torch_df_grouped'] = df_grouped
    torch_particle_dic['max_mom'] = np.max(df[var])
    torch_particle_dic['min_mom']  = np.min(df[var])
    torch_particle_dic['bin_edges']  = bin_edges


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
        input_df = f[cfg.treename].arrays(["B_Tr_T_P", "B_Tr_T_TRUEID"], library="pd")

    torch_df = pd.DataFrame(columns=["momentum", "logSumProton", "logSumKaon", "recoPV"])
    for f in cfg.torch_dlls:
        print(f"Reading TORCH DLLs file: {f}")
        with uproot.open("{}".format(f)) as _f:
            _torch_df= _f["PIDAlgorithm_particle_output"].arrays(["momentum", "logSumProton", "logSumKaon", "recoPV", "truePID"], library="pd")
        torch_df = pd.concat([torch_df, _torch_df], ignore_index = True)

    torch_df["logSumProtonMinusKaon"] = torch_df["logSumProton"] - torch_df["logSumKaon"]
    torch_df["momentum"] = torch_df["momentum"]*1000 # GeV to MeV conversion


    torch_particle_dic = {
        'K': {
            'TRUEID': 321, 'torch_df_grouped': 0., 'max_mom':0., 'min_mom':0., 'bin_edges':0., 'nbins': 500,  'pid_dict_minus': 0.
        },
        'pi': {
            'TRUEID': 211, 'torch_df_grouped': 0., 'max_mom':0., 'min_mom':0., 'bin_edges':0., 'nbins': 2500, 'pid_dict_minus': 0.
        },
        'p': {
            'TRUEID': 2212, 'torch_df_grouped': 0., 'max_mom':0., 'min_mom':0., 'bin_edges':0., 'nbins': 300, 'pid_dict_minus': 0.
        },
        'mu': {
            'TRUEID': 13, 'torch_df_grouped': 0., 'max_mom':0., 'min_mom':0., 'bin_edges':0., 'nbins': 30, 'pid_dict_minus': 0.
        },
        'e': {
            'TRUEID': 11, 'torch_df_grouped': 0., 'max_mom':0., 'min_mom':0., 'bin_edges':0., 'nbins': 30, 'pid_dict_minus': 0.
        },
    }

    for particle in torch_particle_dic.keys():
        create_bins(torch_df, 'momentum', torch_particle_dic[particle])
        torch_particle_dic[particle]['pid_dict_minus'] = torch_particle_dic[particle]['torch_df_grouped'].set_index('momentum_bins')['logSumProtonMinusKaon'].to_dict()
    # torch_grouped_recoPV = create_bins(torch_df, 'recoPV', 10)


    # Apply the function to assign a random PID to each row
    input_df['logSumProtonMinusKaon'] = input_df.apply(lambda row: get_random_pid(row['B_Tr_T_P'], row['B_Tr_T_TRUEID'], torch_particle_dic), axis=1)

    input_df.drop(['B_Tr_T_P', 'B_Tr_T_TRUEID'], axis=1)
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with uproot.recreate(cfg.output) as f:
        f[cfg.treename] = input_df