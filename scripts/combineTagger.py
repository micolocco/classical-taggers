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

def setup_time_vars(time_unit, decay_time_branches, data, dm):
    data["time"] = data[decay_time_branches[0]]
    # data["time_err"] = data[decay_time_branches[1]]

    if time_unit == "fs":
        data["time"] *= 1000
        # data["time_err"] *= 1000
    if time_unit == "c_fs":
        data["time"] *= 1000 / 0.29979
        # data["time_err"] *= 1000 / 0.29979
    if time_unit == "c_ps":
        data["time"] /= 0.29979
        # data["time_err"] /= 0.29979

    data["time_mod_dm"] = data["time"] % (2 * np.pi / dm)
    return data

"""
On MC:
python scripts/combineTagger.py  --tagger OSKaon OSMuon OSElectron SSKaon --decayType Bs2DsPi --run2 --combinationName 'Bs2DsPi MC OS+SS'  --cut allBKGCAT_notSamePV_noOSP_SSK/balanced --tagged_prePath /ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/4_tagged/ --simulation
python scripts/combineTagger.py  --tagger OSKaon OSMuon OSElectron SSPion SSProton --decayType Bd2JpsiKst --run2 --combinationName 'Bd2JpsiKst MC OS+SS'  --cut allBKGCAT_notSamePV_noOSP_SSK/balanced --tagged_prePath /ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/4_tagged/ --simulation

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine the taggers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagged_prePath', help='Folder where the files with the tagging decision are saved', default='/ceph/users/molocco/FlavourTagging/data/reweighted/4_tagged') #/ceph/users/molocco/FlavourTagging/data/reweighted/4_tagged
    parser.add_argument('--tagger', help='List of taggers/single tagger', nargs='+', required=True)
    parser.add_argument('--decayType', help='Decay used for the calibration', type=str)
    #parser.add_argument('--outputPath', help='Name of the output dir', type=str, default='/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024//')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--cut', help='Cut desired', type=str,)
    parser.add_argument('--features', help='Input features used for NN training', default="union_PROBNN") 
    parser.add_argument('--run2', help='If Run2 tagger combination must be computed as well',  action='store_true') # action='store_true' means args.run2 will be set to True if the --run2 argument is provided on the command line.
    parser.add_argument('--combinationName', help='Name used for the output combination', type=str)
    #parser.add_argument('--unbalanced')
    parser.add_argument('--simulation', help='If data are MC or real-data. Used for the calibration',  action='store_true') # action='store_true' means args.simulation will be set to True if the --simulation argument is provided on the command line.
    parser.add_argument('--time-unit', type=str, default="c_ps",
                        help='Unit of the time branches')
    parser.add_argument('--decay-time-branches', type=str, default=["B_DTF_PV_CTAU"], nargs="+",
                        help='Branch names of the decay-time variables (first decay time, second decay-time error).') # Just using decay time for now
    parser.add_argument('--asymmetry_level', help='Asymmetry between B and Bbar with wrong and correct label. asym_level2: asymmetry in training and calibration samples, asym_level1 only calibration, asym_level0 none', default='asym_level1', type=str) 



    print(f'Combining taggers started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg)
    
    #outputPath =f'{cfg.outputPath}/{cfg.decayType}/combinations/'
    if cfg.simulation:
        outputPath =f'{cfg.tagged_prePath}/{cfg.decayType}/combinations/{cfg.cut}_{cfg.features}/{cfg.asymmetry_level}'
    else:
        outputPath =f'{cfg.tagged_prePath}/{cfg.decayType}/block1_combinations/{cfg.cut}_{cfg.features}/{cfg.asymmetry_level}'
    os.makedirs(outputPath, exist_ok=True)
    
    vars = []
    if cfg.simulation:
        B_ID_var = "B_TRUEID"
    else:
        B_ID_var = "B_ID"
        vars.extend(['FillNumber', 'B_DTF_PV_CTAU', 'signal_weights', 'reweighter_weights'])
    vars.extend(['RUNNUMBER', 'EVENTNUMBER', B_ID_var])
    vars.extend(run2_taggers_variables)
    print("Run2 taggers variables:", run2_taggers_variables)
    taggers_dataframes = []  # List to store DataFrames for each tagger
    # Loop over all taggers
    for tagger in cfg.tagger:
        loading_variables = []
        loading_variables = vars + [f'{tagger}_TagDec', f'{tagger}_Eta']
        input_path = os.path.join(cfg.tagged_prePath, cfg.decayType, tagger, cfg.cut, cfg.features, cfg.asymmetry_level,'*.root')
        input_files = glob.glob(input_path)
        print(input_files)
        # Loop over all files
        singleTagger_dataframes = []
        
        for i, f in enumerate(input_files):
            print(f"Reading input file: {f}")
            with uproot.open(f) as _f:
                _df = _f[cfg.treename].arrays(loading_variables, library="pd")
           
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
    print('Dataframe shape must have same row number for a correct combination! Check it!')
    for single_df in taggers_dataframes[1:]:
        print(single_df.shape)
        #df = pd.merge(df, single_df, on=['event_entry', 'B_TRUEID']+run2_taggers_variables, how='outer')   
        if cfg.simulation: df = pd.merge(df, single_df, on=['event_entry', B_ID_var,]+run2_taggers_variables, how='outer')
        else:
            weights =  'reweighter_weights'
            df = pd.merge(df, single_df, on=['event_entry', B_ID_var, "FillNumber", "B_DTF_PV_CTAU"]+ weights+run2_taggers_variables, how='outer')   
        print(f'total:{df.shape}')
        #df = pd.merge(df, single_df, on=['event_entry',], how='outer')   
    print(df.shape)
    class_label_dict = {521: '$B^+$', -521: '$B^-$', 511: '$B^0$', -511: '$\overline{B}^0$', 531: '$B_s^0$', -531: '$\overline{B}_s^0$'}

    runs=['Run3']
    if cfg.run2:
        runs.append('Run2')
    
    if not cfg.simulation:
        if "Bu" in cfg.decayType:
            mode = "Bu"
        elif "Bd" in cfg.decayType:
            mode = "Bd"
        elif "Bs" in cfg.decayType:
            mode = "Bs"
        dm = ft.constants.DeltaM_s if mode != "Bd" else ft.constants.DeltaM_d
        df = setup_time_vars(cfg.time_unit, cfg.decay_time_branches, df, dm)

    for run in runs:
        os.makedirs(f'{outputPath}/{run}', exist_ok=True)
        taggers = ft.TaggerCollection()
        for tagger in cfg.tagger+['Probability_Medium_0_Run2OSVertexCharge']:#['Probability_Medium_0_Run2OSVertexCharge']: #OSVertexCharge (called differently on data tuples as they are more recent)
        #for tagger in cfg.tagger:
           # Adjust name columns
            if tagger == 'Probability_Medium_0_Run2OSVertexCharge':
                eta_column = f'B_Probability_Medium_0_Run2OSVertexCharge_Omega'
                tagDec_column = f'B_Probability_Medium_0_Run2OSVertexCharge_Dec'     
            elif tagger == 'OSVertexCharge':
                eta_column = f'B_Run2_{tagger}_Omega'
                tagDec_column = f'B_Run2_{tagger}_Dec'
            else:
                if run=='Run2':
                    eta_column = f'B_{run}_{tagger}_Omega'
                    tagDec_column = f'B_{run}_{tagger}_Dec'
                else:
                    eta_column = f'{tagger}_Eta'
                    tagDec_column = f'{tagger}_TagDec'
            # Create taggers          
            if cfg.simulation:
                
                taggers.create_tagger(f"{tagger}", 
                                    eta_data =df[eta_column].tolist(), 
                                    dec_data = df[tagDec_column].tolist(), 
                                    B_ID =df[B_ID_var].tolist(), 
                                    mode = 'Bu',)
            else:
                B_ID_var = "B_ID"
                taggers.create_tagger(f"{tagger}", 
                                eta_data =df[eta_column].tolist(), 
                                dec_data = df[tagDec_column].tolist(), 
                                B_ID =df[B_ID_var].tolist(), 
                                mode = mode,
                                #weight=df["reweighter_weights"]. to_numpy().astype(np.float64),
                                weight=df["signal_weights"]. to_numpy().astype(np.float64), 
                                tau_ps=df["time"].to_numpy().astype(np.float64),)

        taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
        taggers.calibrate()
        os.makedirs(f'{outputPath}/{run}/single', exist_ok=True)
        ft.save_calibration(taggers=taggers, title=cfg.combinationName, save_path=f'{outputPath}/{run}/single')
        taggers.plot_calibration_curves(savepath = f'{outputPath}/{run}/single', omega_range="minimal", nbins=10)


        # Combine the taggers into one. With "calibrated=False" we would combine the raw single tagger statistics, which is not usually what we want.
        tagger_combination = taggers.combine_taggers(f'{cfg.combinationName}_{run}', calibrated=True)
        tagger_combination.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
        ## And calibrate this tagger again
        tagger_combination.calibrate()
        ft.save_calibration(taggers=tagger_combination, title=cfg.combinationName, save_path=f'{outputPath}/{run}')
        tagger_combination.plot_calibration_curve(tagger_combination, savepath=f'{outputPath}/{run}')
        scale = (lambda x: x**4, lambda x: x**1/4)
        scale = "linear"
        class_indices = df[B_ID_var].values
        tagger_combination.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
                                            file_name = 'split_calibration_curves.pdf', savepath = f'{outputPath}/{run}', omega_range="minimal", 
                                            nbins = 10, x_scale = scale, y_scale = scale)#, share_y= True, share_x = True)
        print(f'{run} combination created at {outputPath}')

    