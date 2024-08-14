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
    parser.add_argument('--tagged_prePath', help='Folder where the files with tagging decision are saved')
    parser.add_argument('--tagger', help='List of taggers', nargs='+', required=True)
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='.')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--decayType', help='Event decay for calibration', type=str)

    print(f'Combining taggers started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg)
    
    taggers_dataframes = []  # List to store DataFrames for each tagger
    cut ='cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin'
    # Loop over all taggers
    for tagger in cfg.tagger:
        #vars = ['entry', 'RUNNUMBER', 'EVENTNUMBER', f'{tagger}_TagDec', f'{tagger}_Eta', 'B_TRUEID']
        vars = ['event_entry', f'{tagger}_TagDec', f'{tagger}_Eta', 'B_TRUEID']
        #input_path = os.path.join(cfg.tagged_prePath, cfg.decayType, tagger, cut, '*.root')
        #input_files = glob.glob(input_path)
        input_files = [f'{cfg.tagged_prePath}/test.root'] 

        # Loop over all files
        singleTagger_dataframes = []
        for f in input_files:
            print(f"Reading input file: {f}")
            with uproot.open(f) as _f:
                _df = _f[cfg.treename].arrays(vars, library="pd")
            _df.dropna(inplace=True)
            singleTagger_dataframes.append(_df)
        print(f'{pd.concat(singleTagger_dataframes).shape[0]}')
        taggers_dataframes.append(pd.concat(singleTagger_dataframes, ignore_index=True))
        # Merge all DataFrames on the common columns
    
    df = taggers_dataframes[0]
    '''
    for single_df in taggers_dataframes[1:]:
        df = pd.merge(df, single_df, on=['RUNNUMBER', 'EVENTNUMBER', 'B_TRUEID'], how='outer')   
    print(f"Number of events: {df.shape[0]}")

    #removal_time1 = time.time()
    df = utils.remove_multicandidates(df)
    #removal_time2 = round((time.time()- removal_time1) / 60 , 2) 
    #print(f"Removing multicandidates required {removal_time2}s")
    print(f"Number of events: {df.shape[0]}")
    '''
    taggers = ft.TaggerCollection()

    for tagger in cfg.tagger:
        taggers.create_tagger(f"{tagger}", eta_data =df[f'{tagger}_Eta'].tolist(), dec_data = df[f'{tagger}_TagDec'].tolist(), B_ID =df.B_TRUEID.tolist(), mode = 'Bu', )  # Same aruments as Tagger class


    taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    taggers.calibrate()

    # We may need a different function for a specific tagger. This is not possible via the command line.
    #taggers["OSe"].set_calibration(ft.NSplineCalibration(3, ft.link.logit))
    # The following function prints a lot of statistics and tables and calibrates all taggers in the collection.
    #taggers.calibrate()

    # Now we could combine the taggers into one. If we would use "calibrated=False" here we
    # would combine the raw single tagger statistics, which is not usually what we want.
    #tagger_combination = taggers.combine_taggers("MyCombination", calibrated=True)

    # And calibrate this tagger again
   # tagger_combination.calibrate()
    taggers.plot_calibration_curves(savepath = f'{cfg.target_path}', omega_range="minimal", nbins=10)

