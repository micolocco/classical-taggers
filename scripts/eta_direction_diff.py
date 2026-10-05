import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot 
import tqdm
import os
import argparse

import yaml


def read_files_and_apply_selection(files, variables):
    extra_variables = ['file_id', 'RUNNUMBER', 'EVENTNUMBER', 'B_ID']

    df = None
    for file in tqdm.tqdm(files):
        with uproot.open(file) as f:
            df_file = f['DecayTree;1'].arrays(variables + extra_variables, library="pd")
            df_file["event_entry"] =  df_file["file_id"].astype(str) + "_" + df_file["RUNNUMBER"].astype(str) + "_" + df_file["EVENTNUMBER"].astype(str)
            if df is None:
                df = df_file
            else:
                df = pd.concat([df, df_file], ignore_index=True)

    return df


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Script to plot the relation between the predicted mistag proability and the difference"
        " of the track's direction with respect to signal daughter tracks."
    )
    args.add_argument("--input_files", type=str, nargs="+", required=True, help="Tuples containing signal and tagging information.")
    args.add_argument("--output", type=str, required=True, help="Output path of the plot.")
    args.add_argument("--tagger", type=str, required=True, help="Name of the tagger used to predict the mistag probability.")
    args.add_argument("--decayType", type=str, required=True, help="Decay type of the events in the input files.")
    # args.add_argument("--data_type", type=str, required=True, help="Type of data (e.g., 'Data' or 'MC').")
    args = args.parse_args()

    with open('configs/daughter_kinematics.yaml', 'r') as f:
        daughter_kinematics = yaml.safe_load(f)
        kinematics = daughter_kinematics[args.decayType]

    tagger_features = [f'{args.tagger}_Eta', f'{args.tagger}_TagDec', f'{args.tagger}_OMEGA', f'{args.tagger}_CDEC', ]

    df = read_files_and_apply_selection(args.input_files, kinematics + tagger_features)

    prefixes = []
    for kin in kinematics:
        if len(kin.split('_PX')) > 1:
            pre = kin.split('_PX')[0]

            if pre != 'B_Tr_T':
                prefixes.append(pre)

    for pre in prefixes:
        df[f'{pre}_track_diff'] = np.sqrt(
            (df[f'{pre}_PX'] - df[f'B_Tr_T_PX']) ** 2 +
            (df[f'{pre}_PY'] - df[f'B_Tr_T_PY']) ** 2 +
            (df[f'{pre}_PZ'] - df[f'B_Tr_T_PZ']) ** 2
        )

    df['min_track_daughter_diff'] = df[[f'{pre}_track_diff' for pre in prefixes]].min(axis=1)



    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    hist = ax.hist2d(
        df[f'{args.tagger}_Eta'],
        df['min_track_daughter_diff'],
        bins=25,
    )
    fig.colorbar(hist[3],ax=ax, label='Counts')



    plt.xlabel(f"{args.tagger} Eta")
    plt.ylabel("Minimum track-daughter momentum difference")
    plt.savefig(args.output)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    hist = ax.hist2d(
        df[f'{args.tagger}_Eta'],
        df['min_track_daughter_diff'],
        bins=25,
        norm=LogNorm(),
    )
    fig.colorbar(hist[3],ax=ax, label='Counts')



    plt.xlabel(f"{args.tagger} Eta")
    plt.ylabel("Minimum track-daughter momentum difference")
    plt.savefig(args.output.replace(".pdf", "_log.pdf"))