import sys
import numpy as np
import time
import uproot
import pandas as pd
from matplotlib import pyplot as plt
from IPython import embed
import os
import argparse
from pprint import pprint
import lhcb_ftcalib as ft
import datetime
import glob
# Local import
import scripts.pyTorchTraining as pyTrain
from scripts.NNModel import NeuralNetwork
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import utils
from scripts.preSelections import run2_taggers_variables
import psutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine the taggers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagged_prePath', help='Folder where the files with the tagging decision are saved')
    parser.add_argument('--tagger', help='List of taggers/single tagger', nargs='+', required=True)
    parser.add_argument('--decayType', help='Decay used for the calibration', type=str)
    parser.add_argument('--outputPath', help='Name of the output dir', type=str,)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--cut', help='Cut desired', type=str, required=True)
    parser.add_argument('--features', help='Input features used for NN training',) 
    parser.add_argument('--run2', help='If Run2 tagger combination must be computed as well',  action='store_true') # action='store_true' means args.run2 will be set to True if the --Run2 argument is provided on the command line.
    parser.add_argument('--combinationName', help='Name used for the output combination', type=str)
    parser.add_argument('--trained_on', help='Whether Taggers where trained on MC, Data or using domain adaptation', type=str, )

    print(f'Combining taggers started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg)
    
    outputPath =f'{cfg.outputPath}/{cfg.decayType}/combinations/'
    os.makedirs(outputPath, exist_ok=True)

    data_type = 'Data' if 'Data' in cfg.tagged_prePath else 'MC'

    BID = 'B_ID' if data_type == 'Data' else 'B_TRUEID'

    taggers_dataframes = []  # List to store DataFrames for each tagger
    # Loop over all taggers
    for tagger in cfg.tagger:
        vars = run2_taggers_variables + ['RUNNUMBER', 'EVENTNUMBER', f'{tagger}_TagDec', f'{tagger}_Eta', BID]
        input_path = os.path.join(cfg.tagged_prePath, cfg.decayType, tagger, cfg.cut, cfg.features, f'trained_{cfg.trained_on}/*.root')
        print(f'input path: {input_path}')
        input_files = glob.glob(input_path)
        # Loop over all files
        singleTagger_dataframes = []
        
        for i, f in enumerate(input_files):
            print(f"Reading input file {i+1}/{len(input_files)}: {f}", flush=True)
            print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB')
            with uproot.open(f) as _f:
                _df = _f[cfg.treename].arrays(vars, library="pd")
            _df.dropna(inplace=True)
            id = os.path.basename(f)[:-5]
            if id[-7:-2] == '.data':
                id = id[:-7]
            else:
                id = id[:-3]

            _df["event_entry"] = id + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER'], inplace=True)

            singleTagger_dataframes.append(_df)
            
        #print(f'{pd.concat(singleTagger_dataframes).shape[0]}')
        df_merged=pd.concat(singleTagger_dataframes, ignore_index=True)
        df_merged = df_merged.groupby("event_entry").first()
        taggers_dataframes.append(df_merged)
        
        # Merge all DataFrames on the common columns
        
    df = taggers_dataframes[0]
    print(df.shape)
    print(df.columns)
    print('Dataframe shape must have same row number for a correct combination! Chcek it!')
    for single_df in taggers_dataframes[1:]:
        print(single_df.shape)
        df = pd.merge(df, single_df, on=['event_entry', BID]+run2_taggers_variables, how='outer')   
        print(f'total:{df.shape}')

        #df = pd.merge(df, single_df, on=['event_entry',], how='outer')   
    print(df.shape)

    runs=['Run3']
    if cfg.run2:
        runs.append('Run2')

    npar = 3
    for run in runs:
        os.makedirs(f'{outputPath}/{run}', exist_ok=True)
        taggers = ft.TaggerCollection()
        for tagger in cfg.tagger:
            if run=='Run2':
                taggers.create_tagger(f"{tagger}", eta_data =df[f'B_{run}_{tagger}_Omega'].tolist(), dec_data = df[f'B_{run}_{tagger}_Dec'].tolist(), B_ID =df[BID].tolist(), mode = 'Bu', ) 
            else:
                taggers.create_tagger(f"{tagger}", eta_data =df[f'{tagger}_Eta'].tolist(), dec_data = df[f'{tagger}_TagDec'].tolist(), B_ID =df[BID].tolist(), mode = 'Bu', ) 
        taggers.set_calibration(ft.PolynomialCalibration(npar=npar, link=ft.link.logit))
        taggers.calibrate()
        # Combine the taggers into one. With "calibrated=False" we would combine the raw single tagger statistics, which is not usually what we want.
        tagger_combination = taggers.combine_taggers(f'{cfg.combinationName}_{run}', calibrated=True)
        tagger_combination.set_calibration(ft.PolynomialCalibration(npar=npar, link=ft.link.logit))
        ## And calibrate this tagger again
        tagger_combination.calibrate()
        savepath = f'{outputPath}/{run}/trained_{cfg.trained_on}/{cfg.cut}/{cfg.features}'
        taggers.plot_calibration_curves(savepath = savepath, omega_range="minimal", nbins=10)
        ft.plotting.draw_calibration_curve(tagger_combination, savepath=savepath)
        ft.save_calibration(taggers=tagger_combination, title=cfg.combinationName, save_path=savepath)

        class_indices = df[BID].values
        class_label_dict = {521: '$B^+$', -521: '$B^-$'}


        taggers.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
                                                file_name = 'split_calibration_curves.pdf', savepath = f'{savepath}', omega_range="minimal", 
                                                nbins = 10, x_scale = 'linear', y_scale = 'linear')#, share_y= True, share_x = True)
        
        tagger_combination.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
                                                file_name = 'split_calibration_curves.pdf', savepath = f'{savepath}', omega_range="minimal", 
                                                nbins = 10, x_scale = 'linear', y_scale = 'linear')#, share_y= True, share_x = True)

        print(f'{run} combination created at {outputPath}')

    