from scripts.train_tagger import read_files
from scripts.train_tagger import stats_printout
import argparse
import datetime
from rich.console import Console
from rich.table import Table
import numpy as np
import pandas as pd
import os

#Creating control tables

def print_split(df, weighted=False):
    if not weighted:
        df['weights'] = np.ones(len(df))
    else:
        df['weights'] = df['signal_weights']

    BIDs = df[BID].unique()
    trCharges = df["B_Tr_T_Charge"].unique()
    
    total = np.sum(df['weights'])

    console = Console()
    table = Table(show_header=True)

    table.add_column("", justify="left", style='white', width=25)

    for b in BIDs:
        table.add_column(f"BID={b}", justify="left", style='white', width=25)
    
    table.add_column(f"Sum", justify="left", style='white', width=25)

    
    for c in trCharges:
        table.add_row(
            f"Track Charge: {c}",
            "[green]" + str(np.round(np.sum(df[(df[BID] == BIDs[0]) & (df["B_Tr_T_Charge"] == c)]['weights'])/total*100, 1)) + '%',
            "[green]" + str(np.round(np.sum(df[(df[BID] == BIDs[1]) & (df["B_Tr_T_Charge"] == c)]['weights'])/total*100, 1)) + '%',
            str(np.round(np.sum(df[df["B_Tr_T_Charge"] == c]['weights'])/total*100, 1)) + '%'
        )
    
    table.add_section()

    table.add_row(
        f"Sum",
        str(np.round(np.sum(df[df[BID] == BIDs[0]]['weights'])/total*100, 1)) + '%',
        str(np.round(np.sum(df[df[BID] == BIDs[1]]['weights'])/total*100, 1)) + '%',
        str(np.round(np.sum(df['weights'])/total*100, 1)) + '%'
    )


    console.print(table)
    df.drop(columns=['weights'], inplace=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the tagger on the specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--path', help='path of data files',)
    parser.add_argument('--data_type', help="Type of Data used, MC or Data",choices=('MC', 'Data'))
    cfg = parser.parse_args()


    data = [
        os.path.join(cfg.path, f)
        for f in os.listdir(cfg.path)
        if f.endswith('.root')
    ]

    BID = 'B_ID' if  cfg.data_type == 'Data' else 'B_TRUEID'

    vars = [BID, "B_Tr_T_Charge", 'selected']
    if cfg.data_type == 'Data':
        vars.append('signal_weights')


    treename = 'DecayTree;1'# if  cfg.data_type == 'Data' else 'Tuple/DecayTree;1'

    #Reading Data from files
    print(f'Reading of training files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(data)} files.", flush=True)
    # train_df = read_files(cfg.training_data, vars = vars, treename=cfg.treename, augmentation=False, reduce = False, weight_label=None)
    df = read_files(data, vars = vars, treename=treename, augmentation = False, reduce = False, weight_label = None)
    print(f'Reading of training files ends {datetime.datetime.now().strftime("%H:%M:%S")}')

    df = df[df['selected'] == 1]

    if cfg.data_type != 'MC':
        print('Number of Tracks sweighted')
        print_split(df, True)
    print('Number of Tracks unweighted')
    print_split(df, False)

    # df = df.groupby("event_entry").first()


    # if cfg.data_type != 'MC':
    #     print('Number of Events sweighted')
    #     print_split(df, True)
    # else:
    #     print('Number of Events unweighted')
    #     print_split(df, False)




