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


def mark_momentum_overflow_bin(ax, overflow_start):
    """Mark the upper momentum-difference bin as an overflow bin on the y-axis."""
    lower_limit, _ = ax.get_ylim()
    ticks = np.linspace(lower_limit, overflow_start, 6)
    labels = [f'{tick:g}' for tick in ticks[:-1]]
    labels.append(f'≥ {overflow_start:.3g}\n(overflow)')
    ax.set_yticks(ticks, labels=labels)
    ax.set_ylim(lower_limit, overflow_start * 1.05)
    ax.axhline(overflow_start, color='white', linewidth=0.8)


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

    # Reserve the final momentum-difference bin for the upper 5% of values.
    # Eta continues to use its normal 25 bins.
    momentum_diff = df['min_track_daughter_diff'].to_numpy()
    momentum_overflow_start = np.quantile(momentum_diff, 0.95)
    momentum_bin_edges = np.concatenate((
        np.linspace(momentum_diff.min(), momentum_overflow_start, 15),
        [np.nextafter(momentum_diff.max(), np.inf)],
    ))
    histogram_bins = [25, momentum_bin_edges]
    counts, eta_bin_edges, momentum_bin_edges = np.histogram2d(
        df[f'{args.tagger}_Eta'], momentum_diff, bins=histogram_bins,
    )
    column_totals = counts.sum(axis=1, keepdims=True)
    normalized_counts = np.divide(
        counts, column_totals, out=np.zeros_like(counts),
        where=column_totals != 0,
    )



    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    hist = ax.pcolormesh(
        eta_bin_edges,
        momentum_bin_edges,
        normalized_counts.T,
        shading='auto',
    )
    fig.colorbar(hist, ax=ax, label=r'Counts normalized per $\eta$ bin')
    mark_momentum_overflow_bin(ax, momentum_overflow_start)

    ax.set_xlabel(rf"{args.tagger} $\eta$")
    ax.set_ylabel("Minimum track-daughter momentum difference")
    fig.tight_layout()
    plt.savefig(args.output)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    hist = ax.pcolormesh(
        eta_bin_edges,
        momentum_bin_edges,
        np.ma.masked_less_equal(normalized_counts.T, 0),
        shading='auto',
        norm=LogNorm(),
    )
    fig.colorbar(hist, ax=ax, label=r'Counts normalized per $\eta$ bin')
    mark_momentum_overflow_bin(ax, momentum_overflow_start)




    ax.set_xlabel(rf"{args.tagger} $\eta$")
    ax.set_ylabel("Minimum track-daughter momentum difference")
    fig.tight_layout()
    plt.savefig(args.output.replace(".pdf", "_log.pdf"))
