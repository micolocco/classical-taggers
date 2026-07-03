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
ft.constants.propagate_errors = True 
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
    parser.add_argument('--tagger', help='List of taggers/single tagger', nargs='+', required=True)
    parser.add_argument('--decayType', help='Decay used for the calibration', type=str)
    parser.add_argument('--outputPath', help='Name of the output dir', type=str,)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--cut', help='Cut desired, only used for Run3', type=str, required=True)
    parser.add_argument('--features', help='Input features used for NN training, only used for Run3',) 
    parser.add_argument('--run', help='Whether Run3 or Run2 combination should be computed', type=str, choices=('Run3', 'Run2'))
    parser.add_argument('--combinationName', help='Name used for the output combination', type=str)
    parser.add_argument('--data_type', help='Type of data used for calibration: Data or MC', type=str, choices=('Data', 'MC'))
    parser.add_argument('--input_files', help='Path to the input files used for the combination, used to find the data files. tagger_placeholder is used as used as placeholder for the tagger name', type=str)
    parser.add_argument('--calibrations', help='List of calibration json files for each tagger, only used for Run3', nargs='*')

    print(f'Combining taggers started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg)

    BID = 'B_ID' if cfg.data_type == 'Data' else 'B_TRUEID'

    #Convert list of calibrations to dictionary with tagger names as keys
    calibration_dict = {}
    for calibration in cfg.calibrations:
        splits = calibration.split('/')

        #find the first split that starts with SS or OS 
        tagger_name = next((s for s in splits if s.startswith(('SS', 'OS'))),None)
        
        if tagger_name is None:
            raise ValueError(f"Could not find tagger name in calibration path {calibration}. Expected a split starting with 'SS' or 'OS'.")        

        calibration_dict[tagger_name] = calibration
    print(f'Calibration dictionary: {calibration_dict}')



    taggers_dataframes = []  # List to store DataFrames for each tagger
    # Loop over all taggers
    vars = ['file_id', 'RUNNUMBER', 'EVENTNUMBER',  'B_ID'] + run2_taggers_variables
    if cfg.data_type == 'Data':
        vars += ['signal_weights']
        if 'Bu' not in cfg.decayType:
            vars.append('B_TAU')
            if 'Bs' in cfg.decayType:
                vars.append('B_TAUERR')
    else:
        vars += ['B_TRUEID']

    for tagger in cfg.tagger:
        all_vars = vars + [f'{tagger}_TagDec', f'{tagger}_Eta',f'{tagger}_CDEC', f'{tagger}_OMEGA', f'{tagger}_OMEGA_ERR']

        print(f'Processing tagger {tagger}')
        print(f'Base path {cfg.input_files}')
        # input_path = os.path.join(cfg.tagged_prePath, cfg.decayType, tagger, cfg.cut, cfg.features, f'trained_{cfg.trained_on}/*.root')
        input_path = os.path.join(cfg.input_files.replace('tagger_placeholder', tagger), f'*.root')
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
            # _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'file_id'], inplace=True)


            singleTagger_dataframes.append(_df)
        

            
        df_merged=pd.concat(singleTagger_dataframes, ignore_index=True)
        print(df_merged['event_entry'].nunique())
        assert df_merged['event_entry'].nunique() == len(df_merged), f"There are duplicate event entries in the merged DataFrame. Check {f}."

        df_merged = df_merged.groupby("event_entry").first()
        taggers_dataframes.append(df_merged)
        
        # Merge all DataFrames on the common columns
        
    for df in taggers_dataframes:
        print(df.columns)
        print(df.shape)
        print(df.head())
    assert all(taggers_dataframes[0].shape[1] == single_df.shape[1] for single_df in taggers_dataframes), "DataFrames have different number of columns. Check the input files."
    df = taggers_dataframes[0]
    common_columns = ['B_ID', 'RUNNUMBER', 'EVENTNUMBER', 'file_id'] + run2_taggers_variables
    if cfg.data_type == 'Data':
        common_columns += ['signal_weights']
        if 'Bu' not in cfg.decayType:
            common_columns.append('B_TAU')
            if 'Bs' in cfg.decayType:
                common_columns.append('B_TAUERR')
    else:
        common_columns += ['B_TRUEID']
    
    for single_df in taggers_dataframes[1:]:
        df = pd.merge(df, single_df, on=['event_entry']+common_columns, how='outer')
    del taggers_dataframes  # Free memory
    df.reset_index(drop=False, inplace=True)  # Reset index after merging
    print(df.shape)
    print(df.columns)
    print(df.head())

    #Print number of and rows containing NaN values 
    print(f'Number of rows containing NaN values: {df.isna().any(axis=1).sum()}')
    print(f'Number of NaN values in each column:\n{df.isna().sum()}')

    #print head of a dataframe where at least one of columns is NaN
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df[df.isna().any(axis=1)].head())

    assert not df.isna().any().any(), "There are NaN values in the DataFrame. Please check the input files and merging process."



    weights = None
    tau = None
    tau_ps_err = None
    mode = 'Bu'
    if cfg.data_type == 'Data':
        weights = df['signal_weights'].to_numpy()
        if 'Bu' not in cfg.decayType:
            tau = df['B_TAU'].to_numpy()
            mode = cfg.decayType[:2]
            if 'Bs' in cfg.decayType:
                tau_ps_err = df['B_TAUERR'].to_numpy()
        

    npar = 2
    run = cfg.run

    if run == 'Run2':
        taggers = ft.TaggerCollection() 
    else:
        taggers = ft.TargetTaggerCollection()


    for tagger in cfg.tagger:
        if run == 'Run2':
            eta_col = f'B_{run}_{tagger}_Omega'
            dec_col = f'B_{run}_{tagger}_Dec'
            tagger_class = ft.Tagger
        else:
            eta_col = f'{tagger}_Eta'
            dec_col = f'{tagger}_TagDec'
            tagger_class = ft.TargetTagger

        tagger_obj = tagger_class(tagger,
                                    eta_data=df[eta_col], 
                                    dec_data=df[dec_col], 
                                    B_ID=df[BID], 
                                    mode = mode, 
                                    tau_ps=tau, 
                                    tauerr_ps =tau_ps_err,
                                    weight=weights)
        
        if run == 'Run2':
            print(f"Calibrating {tagger}")
            tagger_obj.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
            tagger_obj.calibrate()
            ft.save_calibration(taggers=tagger_obj, title=f"{tagger}.json", save_path=cfg.outputPath)
        else:
            print(f"Loading calibration for {tagger} from {calibration_dict[tagger]}")
            tagger_obj.load(calibration_dict[tagger], tagger_name = tagger, style='delta')
            tagger_obj.apply()
            
        

        tagger_df = tagger_obj.get_dataframe(True)
        print('Calibrated tagging information')
        print(tagger_df.head())
        df[f"{tagger}_CDEC"] = tagger_df[f"{tagger}_CDEC"].values
        df[f"{tagger}_OMEGA"] = tagger_df[f"{tagger}_OMEGA"].values
        df[f"{tagger}_OMEGA_ERR"] = tagger_df[f"{tagger}_OMEGA_ERR"].values

        print(f'{tagger} has loaded tagging power of {tagger_obj.stats.tagging_power(calibrated=True)}')

        taggers.add_taggers(tagger_obj)

        print(f"Checking taggers after addition: {taggers._taggers}")

    target_combination = taggers.combine_taggers(f'{cfg.combinationName}', calibrated=True)

    print(type(target_combination))



    
    uncali_combined_df = target_combination.get_dataframe(calibrated=False) #individual taggers calibrated but not the combination
    tagger_combination = ft.Tagger(f'{cfg.combinationName}',
                                    eta_data=uncali_combined_df[f'{cfg.combinationName}_ETA'].to_numpy(),
                                    dec_data=uncali_combined_df[f'{cfg.combinationName}_DEC'].to_numpy(),
                                    B_ID=df[BID].to_numpy(), 
                                    mode = mode,
                                    tau_ps=tau,
                                    tauerr_ps=tau_ps_err,
                                    weight=weights)
    
    tagger_combination.set_calibration(ft.PolynomialCalibration(npar=npar, link=ft.link.logit))
    ## And calibrate this tagger again
    tagger_combination.calibrate()
    tagger_df = tagger_combination.get_dataframe(True)
    df[f"{cfg.combinationName}_CDEC"] = tagger_df[f"{cfg.combinationName}_CDEC"].values
    df[f"{cfg.combinationName}_OMEGA"] = tagger_df[f"{cfg.combinationName}_OMEGA"].values
    df[f"{cfg.combinationName}_OMEGA_ERR"] = tagger_df[f"{cfg.combinationName}_OMEGA_ERR"].values


    taggers.plot_calibration_curves(savepath = cfg.outputPath, omega_range="minimal", nbins=10)
    ft.plotting.draw_calibration_curve(tagger_combination, savepath=cfg.outputPath)
    ft.save_calibration(taggers=tagger_combination, title=cfg.combinationName, save_path=cfg.outputPath)

    class_indices = df[BID].values
    class_label_dict = {521: '$B^+$', -521: '$B^-$', 511: '$B^0$', -511: r'$\overline{B}^0$'}

    print(type(tagger_combination))
    # taggers.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
    #                                         file_name = 'split_calibration_curves.pdf', savepath = f'{cfg.outputPath}', omega_range="minimal", 
    #                                         nbins = 10, x_scale = 'linear', y_scale = 'linear')
    
    # ft.plotting.draw_split_calibration_curve(tagger_combination, nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
    #                                         file_name = 'split_calibration_curves.pdf', savepath = f'{cfg.outputPath}', omega_range="minimal", 
    #                                         nbins = 10, x_scale = 'linear', y_scale = 'linear')

    print(f'{run} combination created at {cfg.outputPath}')
    print(f"Tagger: {cfg.combinationName}")
    info_dict = {"TaggingEfficiency"     : tagger_combination.stats.tagging_efficiency(calibrated = False),
                "EffectiveMistag"        : tagger_combination.stats.effective_mistag(  calibrated = False),
                "TaggingPower"           : tagger_combination.stats.tagging_power(     calibrated = False),
                "TaggingEfficiency_Cali" : tagger_combination.stats.tagging_efficiency(calibrated = True ), 
                "EffectiveMistag_Cali"   : tagger_combination.stats.effective_mistag(  calibrated = True ), 
                "TaggingPower_Cali"      : tagger_combination.stats.tagging_power(     calibrated = True ),}
    
    # Process the data
    processed_data = {key: pyTrain.propagate_and_round(value, 'Fitpar' not in key) for key, value in info_dict.items()}
    # Format the output
    formatted_data = {key: f"{values[0]} +- {values[1]}" if len(values) > 1 else values[0] for key, values in processed_data.items()}
    formatted_data = {key: f"{values[0]} +- {values[1]}" if len(values) > 1 else values[0] for key, values in processed_data.items()}
    for key, value in formatted_data.items():
        print(f"{key}: {value}")
    print("\n")

    #save the dataframe with all tagging information to root file
    df.drop(columns=['event_entry'], inplace=True)
    with uproot.recreate(f"{cfg.outputPath}/combined_tagged.root") as file:
        file["DecayTree"] = df
    print(f'File with combined tagging information created at {cfg.outputPath}/combined_tagged.root')
        
        
    