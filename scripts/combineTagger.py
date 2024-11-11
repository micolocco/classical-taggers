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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the tagger on the specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagged_prePath', help='Folder where the files with the tagging decision are saved')
    parser.add_argument('--tagger', help='List of taggers/single tagger', nargs='+', required=True)
    parser.add_argument('--decayType', help='Decay used for the calibration', type=str)
    parser.add_argument('--combinationName', help='Name used for the output combination', type=str)
    parser.add_argument('--outputPath', help='Name of the output dir', type=str, default='combinations')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    
    print(f'Combining taggers started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg)
    
    if cfg.outputPath == 'combinations':
        outputPath =f'{cfg.outputPath}/{cfg.decayType}'
        os.makedirs(outputPath, exist_ok=True)
    else:
        outputPath = cfg.outputPath
    
    taggers_dataframes = []  # List to store DataFrames for each tagger
    cut ='cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin'
    # Loop over all taggers
    for tagger in cfg.tagger:
        vars = ['entry', 'RUNNUMBER', 'EVENTNUMBER', f'{tagger}_TagDec', f'{tagger}_Eta', 'B_TRUEID']
        input_path = os.path.join(cfg.tagged_prePath, cfg.decayType, tagger, cut, '*.root')
        input_files = glob.glob(input_path)
        # Loop over all files
        singleTagger_dataframes = []
        for f in input_files:
            print(f"Reading input file: {f}")
            with uproot.open(f) as _f:
                _df = _f[cfg.treename].arrays(vars, library="pd")
            _df.dropna(inplace=True)
            singleTagger_dataframes.append(_df)
        #print(f'{pd.concat(singleTagger_dataframes).shape[0]}')
        taggers_dataframes.append(pd.concat(singleTagger_dataframes, ignore_index=True))
        # Merge all DataFrames on the common columns
    
    df = taggers_dataframes[0]
    
    for single_df in taggers_dataframes[1:]:
        df = pd.merge(df, single_df, on=['RUNNUMBER', 'EVENTNUMBER', 'B_TRUEID'], how='outer')   

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
    taggers.plot_calibration_curves(savepath = f'{outputPath}', omega_range="minimal", nbins=10)
    ft.plotting.draw_calibration_curve(tagger_combination, savepath=f'{outputPath}')
    ft.save_calibration(taggers=tagger_combination, title=combination, save_path=f'{outputPath}')
    print(f'{cfg.combinationName} combination created at {outputPath}')
    