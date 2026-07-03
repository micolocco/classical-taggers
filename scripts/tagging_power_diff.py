import json

import uproot 
import pandas as pd
import os
import numpy as np
import argparse
from pprint import pprint
from rich.console import Console
from rich.table import Table
from rich import box
from io import StringIO
import matplotlib.pyplot as plt

def read_files(files, taggers,labels, tree):
    df = None

    event_id_variables = ['file_id', 'RUNNUMBER', 'EVENTNUMBER']
    general_variables = ['B_ID', 'signal_weights']
    tagging_variables = []
    for tagger in taggers:
        tagging_variables += [f'{tagger}_CDEC', f'{tagger}_OMEGA']

    for f, label in zip(files, labels):
        print(f'Processing {f}')
        with uproot.open(f) as _f:
            _df = _f[tree].arrays(event_id_variables + general_variables + tagging_variables, library="pd")

        # Rename tagging variables to include label for identification
        for tagger in taggers:
            _df.rename(columns={f'{tagger}_CDEC': f'{label}_{tagger}_CDEC', 
                                f'{tagger}_OMEGA': f'{label}_{tagger}_OMEGA', }, inplace=True)
        _df['event_entry'] = _df['file_id'].astype(str) + "_" + _df['RUNNUMBER'].astype(str) + "_" + _df['EVENTNUMBER'].astype(str)

        _df.drop(columns=['file_id', 'RUNNUMBER', 'EVENTNUMBER'], inplace=True)        
        if df is None:
            df = _df
        else:
            df = pd.merge(df, _df, on=general_variables+['event_entry'], how='outer')
    return df


def print_matrix_as_table(matrix, taggers, labels, combination_name, column_width=12, digits=4):
    output = StringIO()
    table_width = len(taggers)*len(labels)*column_width + 50
    console = Console(file=output, width=table_width)
    table = Table(show_header=True, header_style="bold magenta", padding=(0,0))
    table.width = table_width


    table.add_column('', justify="center", width=np.max([len(l) for l in labels]))
    for label in labels:
        table.add_column(label, justify="center", width=column_width*len(taggers))

    def add_super_row(rowTables, i, super_label, include_header=True):
        if i != 0:
                table.add_row(*[super_label] + rowTables)

        rowTables = []
        left_most = True
        for label in labels:
            rowTable = Table(show_header=True, header_style="bold cyan", padding=(0,0), box=box.MINIMAL)
            if left_most:
                rowTable.add_column("", style="dim", width=column_width)
                left_most = False
            for tagger in taggers:
                if tagger == combination_name:
                    header = 'combination'
                else:
                    header = tagger

                if include_header:
                    rowTable.add_column(header, justify="right", width=column_width)
                else:
                    rowTable.add_column(justify="right", width=column_width)
            rowTables.append(rowTable)
        return rowTables

    upper_most = True
    rowTables = []
    for i, data_row in enumerate(matrix):
        if i%len(taggers) == 0:
            rowTables = add_super_row(rowTables, i, labels[i//len(taggers)-1], include_header=upper_most)
            upper_most = False

        left_most = True
        for label_index, label in enumerate(labels):
            tag = taggers[i%len(taggers)]

            row = []
            if tag == combination_name:
                tag = 'combination'
            if left_most:
                row = [tag[:column_width].ljust(column_width)]
                left_most = False
            for j in range(len(taggers)):
                value_str = f"{np.round(data_row[label_index*len(taggers) + j], digits):.{digits}f}"
                row.append(value_str)
            rowTables[label_index].add_row(*row)
        
    add_super_row(rowTables, len(matrix), labels[-1], include_header=False)
    
    console.print(table)
    table_str = output.getvalue()
    print(table_str)
    output.close()

def get_matrix_from_df(df, taggers, labels, target):
    full_taggers = [f'{label}_{tagger}' for label in labels for tagger in taggers]

    matrix = np.zeros((len(full_taggers), len(full_taggers)))
    for i, tag1 in enumerate(full_taggers):
        for j, tag2 in enumerate(full_taggers):
            matrix[i,j] = df.loc[(df['tagger1'] == tag1) & (df['tagger2'] == tag2), target]

    return matrix

def calc_correlation_with_dilution(df, tagger1, tagger2, df_corr=None):
    # Whenever there is an event only getting a decision from one tagger the events are uncorrelated
    #-> get the fraction of events where only one tagger has a decision, and the other does not have a decision (i.e. has a decision of 0)
    frac_uncor = ((df[f'{tagger1}_CDEC'] == 0) ^ (df[f'{tagger2}_CDEC'] == 0)).sum() / ((df[f'{tagger1}_CDEC'] == 1) | (df[f'{tagger2}_CDEC'] == 1)).sum()
    #-> calculate the correlation between the probabilities for the two taggers with decisions
    correlation = df.loc[(df[f'{tagger1}_CDEC'] != 0) & (df[f'{tagger2}_CDEC'] != 0).values, [f'{tagger1}_Prob_B', f'{tagger2}_Prob_B']].corr().iloc[0,1]
    #-> Dilute measured correlation by the fraction of uncorrelated events to get the true correlation between the taggers' decisions
    dil_corr = correlation * (1-frac_uncor)

    frac_opposite = ((df[f'{tagger1}_CDEC'] * df[f'{tagger2}_CDEC']) < 0).sum() / len(df)

    df_corr.loc[len(df_corr)] = [tagger1, tagger2, frac_uncor, frac_opposite, correlation, dil_corr]

def plot_scatter_of_prob(df, tagger1, tagger2, out_path):
    df = df[[f'{tagger1}_CDEC', f'{tagger2}_CDEC', f'{tagger1}_Prob_B', f'{tagger2}_Prob_B']].copy()
    df = df.loc[(df[f'{tagger1}_CDEC'] != 0) & (df[f'{tagger2}_CDEC'] != 0).values]

    plt.scatter(df[f'{tagger1}_Prob_B'], df[f'{tagger2}_Prob_B'], alpha = 0.4, marker='.', s = 1)
    plt.xlabel(r"$P(B^0)$ of " + tagger1.split('_')[0])
    plt.ylabel(r"$P(B^0)$ of " + tagger2.split('_')[0])
    plt.savefig(os.path.join(out_path, f'{tagger1}_{tagger2}_scatter.png'))
    plt.clf()
    print("Scatterplot saved")




'''

Comparison of allBKGCAT_notSamePV_noOSP_SSK_balanced and notSamePV_noOSP
python scripts/tagging_power_diff.py --config_json '{"oldTree": {"combined_data": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root", "SSPion": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/calibration.json", "SSProton": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSKaon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/calibration.json", "OSMuon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "SSPion_SSProton_OSKaon_OSMuon_OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSPion_SSProton_OSKaon_OSMuon_OSElectron.json"}, "newTree": {"combined_data": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root", "SSPion": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/calibration.json", "SSProton": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSKaon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSMuon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "SSPion_SSProton_OSKaon_OSMuon_OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSPion_SSProton_OSKaon_OSMuon_OSElectron.json"}}'

Comparison of current best allBKGCAT_notSamePV_noOSP_SSK_balanced and Run3v1
python scripts/tagging_power_diff.py --config_json '{"currentBest": {"combined_data": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root", "SSPion": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/calibration.json", "SSProton": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSKaon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSMuon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "SSPion_SSProton_OSKaon_OSMuon_OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSPion_SSProton_OSKaon_OSMuon_OSElectron.json"}, "benchmark": {"combined_data": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/with_event_selection/combined_tagged.root", "SSPion": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/SSPion/Run3v1/testing/Data/logit/calibration.json", "SSProton": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/SSProton/Run3v1/testing/Data/logit/calibration.json", "OSKaon": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSKaon/Run3v1/testing/Data/logit/calibration.json", "OSMuon": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSMuon/Run3v1/testing/Data/logit/calibration.json", "OSElectron": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSElectron/Run3v1/testing/Data/logit/calibration.json", "SSPion_SSProton_OSKaon_OSMuon_OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/with_event_selection/SSPion_SSProton_OSKaon_OSMuon_OSElectron.json"}}'

Comparison of allBKGCAT_notSamePV_noOSP_SSK_balanced with and without BN
python scripts/tagging_power_diff.py --config_json '{"noBN": {"combined_data": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root", "SSPion": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/calibration.json", "SSProton": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSKaon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSMuon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json", "SSPion_SSProton_OSKaon_OSMuon_OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSPion_SSProton_OSKaon_OSMuon_OSElectron.json"}, "BN": {"combined_data": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root", "SSPion": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32_BN/testing/Data/logit/calibration.json", "SSProton": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.001_bs8192_nL8_nN128_BN/testing/Data/logit/calibration.json", "OSKaon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.001_bs8192_nL8_nN64_BN/testing/Data/logit/calibration.json", "OSMuon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32_BN/testing/Data/logit/calibration.json", "OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.001_bs8192_nL8_nN32_BN/testing/Data/logit/calibration.json", "SSPion_SSProton_OSKaon_OSMuon_OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSPion_SSProton_OSKaon_OSMuon_OSElectron.json"}}'

Comparison of Run3v0 and Run3v1
python scripts/tagging_power_diff.py --config_json '{"V0": {"combined_data": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root", "SSPion": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSPion.json", "SSProton": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSProton.json", "OSKaon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/OSKaon.json", "OSMuon": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/OSMuon.json", "OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/OSElectron.json", "SSPion_SSProton_OSKaon_OSMuon_OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSPion_SSProton_OSKaon_OSMuon_OSElectron.json"}, "V1": {"combined_data": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root", "SSPion": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/SSPion/Run3v1/testing/Data/logit/calibration.json", "SSProton": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/SSProton/Run3v1/testing/Data/logit/calibration.json", "OSKaon": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSKaon/Run3v1/testing/Data/logit/calibration.json", "OSMuon": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSMuon/Run3v1/testing/Data/logit/calibration.json", "OSElectron": "/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSElectron/Run3v1/testing/Data/logit/calibration.json", "SSPion_SSProton_OSKaon_OSMuon_OSElectron": "/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/SSPion_SSProton_OSKaon_OSMuon_OSElectron.json"}}'

'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to estimate correlation between tagging powers and compute the tagging power difference between two taggers.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config_json', type=str, help='JSON string describing inputs. Format: {"label1": {"combined_data": "path", "tagger1": "path_toCalib_info", "tagger2": "path_toCalib_info", ..., "combination_name": "path_toCalib_info"}, "label2_toCalib_info": {...}}. The combination tagger (last key) will be used for correlation calculations.', required=True)
    parser.add_argument('--tree', type=str, default='DecayTree;1', help='Name of the tree in the ROOT files to read.')
    parser.add_argument('--outpath', type=str, help='Path to where the output should be saved', default='/ceph/users/togasa/FlavourTagging/comparisons/')
    parser.add_argument('--use_uncalibrated', action='store_true', help='Whether to use the calibrated probabilities for the correlation calculations.')
    cfg = parser.parse_args()

    #make sure outpath exist
    os.makedirs(cfg.outpath, exist_ok=True)


    config = json.loads(cfg.config_json)
    labels = list(config.keys())

    first_entry = config[labels[0]]
    taggers = [k for k in first_entry.keys() if k != 'combined_data']


    tagger_calibs = {}
    for label in labels:
        for tagger in taggers:
            tagger_calibs[f'{label}_{tagger}'] = config[label][tagger]

    combined_files = []
    calibration_info = []
    for label in labels:
        entry = config[label]
        combined_files.append(entry['combined_data'])
        for t in taggers:
            if t in entry:
                calibration_info.append(entry[t])
            else:
                raise ValueError(f"Missing calibration info for tagger {t} under label {label}")


    tagger_list = taggers#[:-1]
    combination_name = taggers[-1]

    pprint({'labels': labels, 'taggers': tagger_list, 'combination': combination_name})

    df = read_files(combined_files, tagger_list, labels, cfg.tree)

    assert df['event_entry'].nunique() == len(df), "There are duplicate event entries in the data. Investigate."

    print(df.head())
    print(df.columns)

    # convert tagdec and mistag to probability of being B or anti-B
    # calculate the correlation between the probabilities for each pair of taggers, and print in a table format

    for label in labels:
        for tagger in tagger_list:
            df[f'{label}_{tagger}_Prob_B'] = (1+df[f'{label}_{tagger}_CDEC'])/2
            df.loc[df[f'{label}_{tagger}_Prob_B'] == 1, f'{label}_{tagger}_Prob_B'] -= df.loc[df[f'{label}_{tagger}_Prob_B'] == 1, f'{label}_{tagger}_OMEGA']
            df.loc[df[f'{label}_{tagger}_Prob_B'] == 0, f'{label}_{tagger}_Prob_B'] += df.loc[df[f'{label}_{tagger}_Prob_B'] == 0, f'{label}_{tagger}_OMEGA']




    vars_to_check = [f'{label}_{tagger}' for label in labels for tagger in tagger_list]
    df_corr = pd.DataFrame(columns=['tagger1', 'tagger2', 'frac_uncorrelated', 'frac_opposite', 'correlation', 'diluted_correlation'])
    for i, var1 in enumerate(vars_to_check):
        for j, var2 in enumerate(vars_to_check):
            calc_correlation_with_dilution(df, var1, var2, df_corr)


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df_corr)

    print("Fraction of uncorrelated decision events between taggers:")
    print_matrix_as_table(get_matrix_from_df(df_corr, tagger_list, labels, 'frac_uncorrelated'), tagger_list, labels, combination_name=combination_name)

    print("Fraction of opposite decision events between taggers:")
    print_matrix_as_table(get_matrix_from_df(df_corr, tagger_list, labels, 'frac_opposite'), tagger_list, labels, combination_name=combination_name)

    print("Raw correlation between taggers' probabilities of being B:")
    print_matrix_as_table(get_matrix_from_df(df_corr, tagger_list, labels, 'correlation'), tagger_list, labels, combination_name=combination_name)

    print("Diluted correlation between taggers' probabilities of being B:")
    print_matrix_as_table(get_matrix_from_df(df_corr, tagger_list, labels, 'diluted_correlation'), tagger_list, labels, combination_name=combination_name)



    # Load the tagging powers for each tagger and combination, and compute the differences in tagging power between each combination 
    # with error propagation

    df_tagging_power = pd.DataFrame(columns=['tagger', 'tagging_power', 'stat_unc', 'calib_unc'])

    print(config)

    for tagger, calib_info in tagger_calibs.items():
        print(f"Loading calibration info for {tagger} from {calib_info}")


        with open(calib_info, 'r') as f:
            calib_data = json.load(f)

            # Drop the label
            tagger_name = tagger[len(tagger.split('_')[0])+1:]


            if tagger_name not in calib_data:
                tagger_name = tagger_name + '_Run3'
                
            tagging_power = calib_data[tagger_name]['calibrated']['overall']['tagging_power']


            df_tagging_power.loc[len(df_tagging_power)] = [tagger, tagging_power[0], tagging_power[1], tagging_power[2]]

    print(df_tagging_power)

            
    for tagger in tagger_list:
        tagger1 = f'{labels[0]}_{tagger}'
        tagger2 = f'{labels[1]}_{tagger}'

        tp1 = df_tagging_power.loc[df_tagging_power['tagger'] == tagger1, 'tagging_power']
        tp2 = df_tagging_power.loc[df_tagging_power['tagger'] == tagger2, 'tagging_power']

        assert len(tp1) == 1 and len(tp2) == 1, f"Tagging power for {tagger1} or {tagger2} not found or not unique in the tagging power dataframe. Check the tagging power dataframe:\n{df_tagging_power}"
        tp1 = tp1.values[0]
        tp2 = tp2.values[0]
        tp_diff = (tp1 - tp2)*100

        print(f"{tagger}: {tp1:.4f} vs {tp2:.4f}, difference: {tp_diff:.4f}%")

        stat_unc1 = df_tagging_power.loc[df_tagging_power['tagger'] == tagger1, 'stat_unc']
        stat_unc2 = df_tagging_power.loc[df_tagging_power['tagger'] == tagger2, 'stat_unc']
        assert len(stat_unc1) == 1 and len(stat_unc2) == 1, f"Statistical uncertainty for {tagger1} or {tagger2} not found or not unique in the tagging power dataframe. Check the tagging power dataframe:\n{df_tagging_power}"
        stat_unc1 = stat_unc1.values[0]
        stat_unc2 = stat_unc2.values[0]

        # Correlation of statistical uncertainty is 100% 
        stat_unc_diff = np.sqrt(stat_unc1**2 + stat_unc2**2 - 2*stat_unc1*stat_unc2)*100

        calib_unc1 = df_tagging_power.loc[df_tagging_power['tagger'] == tagger1, 'calib_unc']
        calib_unc2 = df_tagging_power.loc[df_tagging_power['tagger'] == tagger2, 'calib_unc']
        assert len(calib_unc1) == 1 and len(calib_unc2) == 1, f"calibration uncertainty for {tagger1} or {tagger2} not found or not unique in the tagging power dataframe. Check the tagging power dataframe:\n{df_tagging_power}"
        calib_unc1 = calib_unc1.values[0]
        calib_unc2 = calib_unc2.values[0]

        # Correlation is diluted correlation between taggers' probabilities of being B
        corr_calib_unc = df_corr.loc[(df_corr['tagger1'] == tagger1) & (df_corr['tagger2'] == tagger2), 'diluted_correlation']
        assert len(corr_calib_unc) == 1, f"Diluted correlation for {tagger1} and {tagger2} not found or not unique in the correlation dataframe. Check the correlation dataframe:\n{df_corr}"
        corr_calib_unc = corr_calib_unc.values[0]
        calib_unc_diff = np.sqrt(calib_unc1**2 + calib_unc2**2 - 2*corr_calib_unc*calib_unc1*calib_unc2)*100

        df_tagging_power.loc[len(df_tagging_power)] = ["diff_" + tagger, tp_diff, stat_unc_diff, calib_unc_diff]


        print(f"Tagger: {tagger}")
        print(f"statistical correlation: {1}, calibration correlation: {corr_calib_unc:.4f}")
        print(f"Order: {labels[0]} - {labels[1]}")
        print(f"Tagging power difference: ({tp_diff:.4f} +/- {np.sqrt(stat_unc_diff**2 + calib_unc_diff**2):.4f})% [{stat_unc_diff:.4f}% (stat), {calib_unc_diff:.4f}% (calib)] ")

    out_path = os.path.join(cfg.outpath, f'{labels[0]}_vs_{labels[1]}')
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, 'tagging_power_comparison.csv'), 'w') as f:
        df_tagging_power.to_csv(f, index=False)
    with open(os.path.join(out_path, 'tagger_correlations.csv'), 'w') as f:
        df_corr.to_csv(f, index=False)


    plot_scatter_of_prob(df, 'V0_SSPion_SSProton_OSKaon_OSMuon_OSElectron', 'V1_SSPion_SSProton_OSKaon_OSMuon_OSElectron', out_path)
    plot_scatter_of_prob(df, 'V0_OSKaon', 'V1_OSKaon', out_path)
    plot_scatter_of_prob(df, 'V0_OSMuon', 'V1_OSMuon', out_path)
    plot_scatter_of_prob(df, 'V0_OSElectron', 'V1_OSElectron', out_path)
    plot_scatter_of_prob(df, 'V0_SSPion', 'V1_SSPion', out_path)
    plot_scatter_of_prob(df, 'V0_SSProton', 'V1_SSProton', out_path)
