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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine the taggers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagged_prePath', help='Folder where the files with the tagging decision are saved', default='/ceph/users/molocco/Data/withUT_MC_2024/4_tagged')
    parser.add_argument('--tagger', help='List of taggers/single tagger', nargs='+', required=True)
    parser.add_argument('--decayType', help='Decay used for the calibration', type=str)
    parser.add_argument('--combinationName', help='Name used for the output combination', type=str)
    parser.add_argument('--outputPath', help='Name of the output dir', type=str, default='/ceph/users/molocco/Data/savedModels/withUT_MC_2024/')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--cut', help='Cut desired', type=str, required=True)
    parser.add_argument('--features', help='Input features used for NN training', default='tagger_inputFeatures/union') 

    
    print(f'Combining taggers started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg)
    
    
    outputPath =f'{cfg.outputPath}/{cfg.decayType}/combinations/'
    os.makedirs(outputPath, exist_ok=True)
    os.makedirs(f'{outputPath}/run2', exist_ok=True)
    os.makedirs(f'{outputPath}/run3', exist_ok=True)
    
    taggers_dataframes = []  # List to store DataFrames for each tagger
    # Loop over all taggers
    for tagger in cfg.tagger:
        vars = run2_taggers_variables + ['RUNNUMBER', 'EVENTNUMBER', f'{tagger}_TagDec', f'{tagger}_Eta', 'B_TRUEID']
        input_path = os.path.join(cfg.tagged_prePath, cfg.decayType, tagger, cfg.cut, cfg.features, '*.root')
        input_files = glob.glob(input_path)
        print(input_path)
        # Loop over all files
        singleTagger_dataframes = []
        
        for i, f in enumerate(input_files):
            print(f"Reading input file: {f}")
            with uproot.open(f) as _f:
                _df = _f[cfg.treename].arrays(vars, library="pd")
            _df.dropna(inplace=True)
            _df["SAMPLENUMBER"] = i
            _df["event_entry"] = _df["SAMPLENUMBER"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'SAMPLENUMBER'], inplace=True)

            singleTagger_dataframes.append(_df)
            
        #print(f'{pd.concat(singleTagger_dataframes).shape[0]}')
        df_merged=pd.concat(singleTagger_dataframes, ignore_index=True)
        df_merged = df_merged.groupby("event_entry").first()
        taggers_dataframes.append(df_merged)
        
        # Merge all DataFrames on the common columns
        
    df = taggers_dataframes[0]
    print(df.shape)
    print(df.columns)
    for single_df in taggers_dataframes[1:]:
        print(single_df.shape)
        df = pd.merge(df, single_df, on=['event_entry', 'B_TRUEID']+run2_taggers_variables, how='outer')   
        print(f'total:{df.shape}')

        #df = pd.merge(df, single_df, on=['event_entry',], how='outer')   
    print(df.shape)

    
    taggers = ft.TaggerCollection()

    for tagger in cfg.tagger:
        taggers.create_tagger(f"{tagger}", eta_data =df[f'{tagger}_Eta'].tolist(), dec_data = df[f'{tagger}_TagDec'].tolist(), B_ID =df.B_TRUEID.tolist(), mode = 'Bu', ) 

    taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    taggers.calibrate()

    # Combine the taggers into one. With "calibrated=False" we would combine the raw single tagger statistics, which is not usually what we want.
    combination = cfg.combinationName
    tagger_combination = taggers.combine_taggers(combination, calibrated=True)
    tagger_combination.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    ## And calibrate this tagger again
    tagger_combination.calibrate()
    taggers.plot_calibration_curves(savepath = f'{outputPath}/run3', omega_range="minimal", nbins=10)
    ft.plotting.draw_calibration_curve(tagger_combination, savepath=f'{outputPath}/run3')
    ft.save_calibration(taggers=tagger_combination, title=combination, save_path=f'{outputPath}/run3')
    
    run2_taggers = ft.TaggerCollection()
    for tagger in cfg.tagger:
        run2_taggers.create_tagger(f"{tagger}", eta_data =df[f'B_Run2_{tagger}_Omega'].tolist(), dec_data = df[f'B_Run2_{tagger}_Dec'].tolist(), B_ID =df.B_TRUEID.tolist(), mode = 'Bu', ) 

    run2_taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    run2_taggers.calibrate()

    # Combine the taggers into one. With "calibrated=False" we would combine the raw single tagger statistics, which is not usually what we want.
    combination = cfg.combinationName
    run2_tagger_combination = run2_taggers.combine_taggers(combination, calibrated=True)
    run2_tagger_combination.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    ## And calibrate this tagger again
    run2_tagger_combination.calibrate()
    run2_taggers.plot_calibration_curves(savepath = f'{outputPath}/run2', omega_range="minimal", nbins=10)
    ft.plotting.draw_calibration_curve(run2_tagger_combination, savepath=f'{outputPath}/run2')
    ft.save_calibration(taggers=run2_tagger_combination, title=combination, save_path=f'{outputPath}/run2')
    
    print(f'{cfg.combinationName} combination created at {outputPath}')
    