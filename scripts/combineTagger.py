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
    parser.add_argument('--data_type', help='Type of data used for calibration: Data or MC', type=str, choices=('Data', 'MC'))
    parser.add_argument('--trained_on', help='Whether Taggers where trained on MC, Data or using domain adaptation', type=str, )

    print(f'Combining taggers started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg)
    
    outputPath =f'{cfg.outputPath}/{cfg.decayType}/combinations/'
    os.makedirs(outputPath, exist_ok=True)



    taggers_dataframes = []  # List to store DataFrames for each tagger
    # Loop over all taggers
    vars = run2_taggers_variables + ['file_id', 'RUNNUMBER', 'EVENTNUMBER',  'B_ID']
    if cfg.data_type == 'Data':
        vars += ['signal_weights']
        if 'Bu' not in cfg.decayType:
            vars.append('B_TAU')
    for tagger in cfg.tagger:
        all_vars = vars + [f'{tagger}_TagDec', f'{tagger}_Eta']

        input_path = os.path.join(cfg.tagged_prePath, cfg.decayType, tagger, cfg.cut, cfg.features, f'trained_{cfg.trained_on}/*.root')
        print(f'input path: {input_path}')
        input_files = glob.glob(input_path)
        # Loop over all files
        singleTagger_dataframes = []

        
        for i, f in enumerate(input_files):
            print(f"Reading input file {i+1}/{len(input_files)}: {f}", flush=True)
            print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB')
            with uproot.open(f) as _f:
                _df = _f[cfg.treename].arrays(all_vars, library="pd")
            _df.dropna(inplace=True)

            _df["event_entry"] = _df["file_id"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'file_id'], inplace=True)


            singleTagger_dataframes.append(_df)
        

            
        df_merged=pd.concat(singleTagger_dataframes, ignore_index=True)
        print(df_merged['event_entry'].nunique())
        df_merged = df_merged.groupby("event_entry").first()
        taggers_dataframes.append(df_merged)
        
        # Merge all DataFrames on the common columns
        
    for df in taggers_dataframes:
        print(df.columns)
        print(df.shape)
        print(df.head())
    assert all(taggers_dataframes[0].shape[1] == single_df.shape[1] for single_df in taggers_dataframes), "DataFrames have different number of columns. Check the input files."
    df = taggers_dataframes[0]
    merge_columns = ['event_entry','B_ID']+run2_taggers_variables
    if cfg.data_type == 'Data':
        merge_columns += ['signal_weights']
        if 'Bu' not in cfg.decayType:
            merge_columns.append('B_TAU')
    
    for single_df in taggers_dataframes[1:]:
        df = pd.merge(df, single_df, on=merge_columns, how='inner')
    del taggers_dataframes  # Free memory
    print(df.shape)
    print(df.columns)
    print(df.head())

    runs=['Run3']
    if cfg.run2:
        runs.append('Run2')


    weights = None
    tau = None
    mode = 'Bu'
    if cfg.data_type == 'Data':
        weights = df['signal_weights'].to_numpy()
        if 'Bu' not in cfg.decayType:
            tau = df['B_TAU'].to_numpy()
            mode = cfg.decayType[:2]


    


    npar = 2
    for run in runs:
        os.makedirs(f'{outputPath}/{run}', exist_ok=True)
        taggers = ft.TaggerCollection()
        for tagger in cfg.tagger:
            print(f'Creating tagger: {tagger}')
            print(f'{tagger}_Eta')
            print(len(df[df[f'{tagger}_TagDec'] == 0]))
            print(len(df[df[f'{tagger}_TagDec'] != 0]))
            

            if run == 'Run2':
                eta_col = f'B_{run}_{tagger}_Omega'
                dec_col = f'B_{run}_{tagger}_Dec'
            else:
                eta_col = f'{tagger}_Eta'
                dec_col = f'{tagger}_TagDec'

            print(df[eta_col].head())
            taggers.create_tagger(f"{tagger}", 
                                    eta_data =df[eta_col].to_numpy(), 
                                    dec_data = df[dec_col].to_numpy(), 
                                    B_ID =df['B_ID'].to_numpy(), 
                                    mode = mode, 
                                    weight = weights,
                                    tau_ps = tau)
            
        taggers.set_calibration(ft.PolynomialCalibration(npar=npar, link=ft.link.logit))
        taggers.calibrate()
        # Combine the taggers into one. With "calibrated=False" we would combine the raw single tagger statistics, which is not usually what we want.
        tagger_combination = taggers.combine_taggers(f'{cfg.combinationName}_{run}', calibrated=True)
        tagger_combination.set_calibration(ft.PolynomialCalibration(npar=npar, link=ft.link.logit))
        ## And calibrate this tagger again
        tagger_combination.calibrate()
        savepath = f'{outputPath}/{run}/trained_{cfg.trained_on}/{cfg.cut}/{cfg.features}/{cfg.combinationName}'
        taggers.plot_calibration_curves(savepath = savepath, omega_range="minimal", nbins=10)
        ft.plotting.draw_calibration_curve(tagger_combination, savepath=savepath)
        ft.save_calibration(taggers=tagger_combination, title=cfg.combinationName, save_path=savepath)

        class_indices = df['B_ID'].values
        class_label_dict = {521: '$B^+$', -521: '$B^-$', 511: '$B^0$', -511: '$\overline{B}^0$'}

        print(type(tagger_combination))
        taggers.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
                                                file_name = 'split_calibration_curves.pdf', savepath = f'{savepath}', omega_range="minimal", 
                                                nbins = 10, x_scale = 'linear', y_scale = 'linear')
        
        ft.plotting.draw_split_calibration_curve(tagger_combination, nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
                                                file_name = 'split_calibration_curves.pdf', savepath = f'{savepath}', omega_range="minimal", 
                                                nbins = 10, x_scale = 'linear', y_scale = 'linear')

        print(f'{run} combination created at {outputPath}')

        #Print tagging statistics in human readable format
        for tagger in cfg.tagger:
            print(f"Tagger: {tagger}")
            info_dict = {"TaggingEfficiency"     : taggers[tagger].stats.tagging_efficiency(calibrated = False),
                        "TaggingPower"           : taggers[tagger].stats.tagging_power(     calibrated = False),
                        "EffectiveMistag"        : taggers[tagger].stats.effective_mistag(  calibrated = False),
                        "TaggingEfficiency_Cali" : taggers[tagger].stats.tagging_efficiency(calibrated = True ), 
                        "TaggingPower_Cali"      : taggers[tagger].stats.tagging_power(     calibrated = True ),
                        "EffectiveMistag_Cali"   : taggers[tagger].stats.effective_mistag(  calibrated = True ),} 
            
            # Process the data
            processed_data = {key: pyTrain.propagate_and_round(value, 'Fitpar' not in key) for key, value in info_dict.items()}
            # Format the output
            formatted_data = {key: f"{values[0]} +- {values[1]}" if len(values) > 1 else values[0] for key, values in processed_data.items()}
            formatted_data = {key: f"{values[0]} +- {values[1]}" if len(values) > 1 else values[0] for key, values in processed_data.items()}
            for key, value in formatted_data.items():
                print(f"{key}: {value}")
            print("\n")
        
        print(f"Tagger: {cfg.combinationName}")
        info_dict = {"TaggingEfficiency"     : tagger_combination.stats.tagging_efficiency(calibrated = False),
                    "TaggingPower"           : tagger_combination.stats.tagging_power(     calibrated = False),
                    "EffectiveMistag"        : tagger_combination.stats.effective_mistag(  calibrated = False),
                    "TaggingEfficiency_Cali" : tagger_combination.stats.tagging_efficiency(calibrated = True ), 
                    "TaggingPower_Cali"      : tagger_combination.stats.tagging_power(     calibrated = True ),
                    "EffectiveMistag_Cali"   : tagger_combination.stats.effective_mistag(  calibrated = True ),} 
        
        # Process the data
        processed_data = {key: pyTrain.propagate_and_round(value, 'Fitpar' not in key) for key, value in info_dict.items()}
        # Format the output
        formatted_data = {key: f"{values[0]} +- {values[1]}" if len(values) > 1 else values[0] for key, values in processed_data.items()}
        formatted_data = {key: f"{values[0]} +- {values[1]}" if len(values) > 1 else values[0] for key, values in processed_data.items()}
        for key, value in formatted_data.items():
            print(f"{key}: {value}")
        print("\n")
        
        
    