import lhcb_ftcalib as ft
import os
import argparse
from pprint import pprint
import uproot
import numpy as np
from IPython import embed
import datetime
import matplotlib.pyplot as plt
from scripts import matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
""" 
python data_calibration/data_calib.py  --tagger OSKaon OSMuon OSElectron SSPion SSProton  --decayType Bd2JpsiKst --run2 --combinationName 'B0 Data, OS+SS'  --cut allBKGCAT_notSamePV_noOSP_SSK --block 1 | tee /ceph/users/molocco/FlavourTagging/data_calibration/Bd2JpsiKst/allBKGCAT_notSamePV_noOSP_SSK/block1_asym_level1/calib_log.log
python data_calibration/data_calib.py  --tagger OSKaon OSMuon OSElectron  --decayType Bu2JpsiK --run2 --combinationName 'B+ Data, OS'  --cut allBKGCAT_notSamePV_noOSP_SSK --block 1 | tee /ceph/users/molocco/FlavourTagging/data_calibration/Bu2JpsiK/allBKGCAT_notSamePV_noOSP_SSK/block1_asym_level1/calib_log.log
"""
def setup_time_vars(time_unit, decay_time_branches, data, dm):
    data["time"] = data[decay_time_branches[0]]
   #data["time_err"] = data[decay_time_branches[1]]

    if time_unit == "fs":
        data["time"] *= 1000
        #data["time_err"] *= 1000
    if time_unit == "c_fs":
        data["time"] *= 1000 / 0.29979
       # data["time_err"] *= 1000 / 0.29979
    if time_unit == "c_ps":
        data["time"] /= 0.29979
       # data["time_err"] /= 0.29979

    data["time_mod_dm"] = data["time"] % (2 * np.pi / dm) #Projecting decay time to 1 oscillation decay time period
    #data["time_mod_dm_err"] = data["time_err"] % (2 * np.pi / dm)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine the taggers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input_path', help='Folder where tuple with sweights is stored. Used to build the path for the output directory as well', default='/ceph/users/molocco/FlavourTagging/data_calibration')
    parser.add_argument('--tagger', help='List of taggers/single tagger', nargs='+', required=True)
    parser.add_argument('--decayType', help='Decay used for the calibration', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--cut', help='Cut used', type=str, required=True)
    #parser.add_argument('--features', help='Input features used for NN training',) 
    parser.add_argument('--run2', help='If Run2 tagger combination must be computed as well',  action='store_true') # action='store_true' means args.run2 will be set to True if the --Run2 argument is provided on the command line.
    parser.add_argument('--combinationName', help='Name used for the output combination', type=str)
    #parser.add_argument('--output_folder', help='Name of the output dir', type=str,default='block1_data_combinations_fix')
    parser.add_argument('--time-unit', type=str, default="c_ps",
                        help='Unit of the time branches')
    parser.add_argument('--decay-time-branches', type=str, default=["B_DTF_PV_CTAU"], nargs="+", #"B_DTF_PV_CTAUERR"
                        help='Branches names of the decay-time variables (first decay time, second decay-time error).') # Just using decay time for now
    parser.add_argument('--block', help='Block used', type=str, default='1', required=True, choices=['1', '2', '3', '2_3', '1_2', 'all'])
    parser.add_argument('--asym', help='Asymmetry level used', type=str, default='asym_level1',)


    
    cfg = parser.parse_args()
    pprint(cfg)

    print(f'Started at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    path =f'{cfg.input_path}/{cfg.decayType}/{cfg.cut}'
    outputPath = f'{path}/block{cfg.block}_{cfg.asym}/combinations/'
    os.makedirs(outputPath, exist_ok=True)
    links = ['logit', 'mistag']

    B_ID_var = "B_ID"
    input_file = f'{path}/block{cfg.block}_{cfg.asym}/data_fit/sweights.root' # This NTuple is produced by the sWeights.py script. sweights.py run over the ntuples with the attached tagging decision and attach the sweights
    print(f'Loading data from {input_file}')
    with uproot.open(input_file) as f:
        df = f['DecayTree'].arrays(library="pd").reset_index(drop=True)
    
    runs=['Run3']
    if cfg.run2:
        runs.append('Run2')

    # run = "Run3"
    if "Bu" in cfg.decayType:
        mode = "Bu"
    elif "Bd" in cfg.decayType:
        mode = "Bd"
    elif "Bs" in cfg.decayType:
        mode = "Bs"

    dm = ft.constants.DeltaM_s if mode != "Bd" else ft.constants.DeltaM_d
    df = setup_time_vars(cfg.time_unit, cfg.decay_time_branches, df, dm)
    print(mode)
    '''
    # Sara's code
    taggers = ft.TaggerCollection()
    i=0
    run = "Run2" if "Run2" in cfg.tagger[0] else "Run3"
    for tagger in cfg.tagger:
        if run == "Run3":
            taggers.create_tagger(f"{tagger}",
                                  eta_data =df[f'{tagger}_Eta'].to_numpy(),
                                  dec_data = df[f'{tagger}_TagDec'].to_numpy(),
                                  B_ID =df[B_ID_var].to_numpy(),
                                  mode = mode,
                                  weight=df["signal_weights"].to_numpy().astype(np.float64), 
                                  tau_ps=df["time"].to_numpy().astype(np.float64),)
        else:
            taggers.create_tagger(f"{tagger}",
                                  eta_data =df[f'{tagger}_Omega'].to_numpy(),
                                  dec_data = df[f'{tagger}_Dec'].to_numpy(),
                                  B_ID =df[B_ID_var].to_numpy(),
                                  mode = mode,
                                  weight=df["signal_weights"].to_numpy().astype(np.float64),
                                  tau_ps=df["time"].to_numpy().astype(np.float64),)
        # Different calibration curves for each tagger, maybe put a if wrt to tagger name
        if i==0:
            # taggers[i].set_calibration(ft.PolynomialCalibration(npar=3, link=ft.link.logit))
            taggers[i].set_calibration(ft.PolynomialCalibration(npar=3, link=ft.link.logit))
        else:
            taggers[i].set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
        i+=1
    # taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    taggers.calibrate()
    # Combine the taggers into one. With "calibrated=False" we would combine the raw single tagger statistics, which is not usually what we want.
    tagger_combination = taggers.combine_taggers(f'{cfg.combinationName}_{run}', calibrated=True)
    tagger_combination.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    ## And calibrate this tagger again
    tagger_combination.calibrate()
    taggers.plot_calibration_curves(savepath = f'{outputPath}', omega_range="minimal", nbins=10)
    ft.plotting.draw_calibration_curve(tagger_combination, savepath=f'{outputPath}', nbins=20)
    ft.save_calibration(taggers=tagger_combination, title=cfg.combinationName, save_path=f'{outputPath}')
    print(f'{run} combination created at {outputPath}')

    '''
    for run in runs:
        for link in links:
            os.makedirs(f'{outputPath}/{run}/{link}', exist_ok=True)
            tagger_collection = ft.TaggerCollection()
            for i, tagger in enumerate(cfg.tagger):
                if run == "Run2":
                    eta_column = f'B_{run}_{tagger}_Omega'
                    tagDec_column = f'B_{run}_{tagger}_Dec' 
                    print(eta_column)   
                else:
                    #if tagger=='SSProton' or tagger=='SSPion' or tagger=='OSKaon':
                    #if tagger=='OSKaon':
                    #    eta_column = f'B_Run2_{tagger}_Omega'
                    #    tagDec_column = f'B_Run2_{tagger}_Dec'
                    #else:
                    #    eta_column = f'{tagger}_Eta'
                    #    tagDec_column = f'{tagger}_TagDec'
                    eta_column = f'{tagger}_Eta'
                    tagDec_column = f'{tagger}_TagDec'
                tagger_collection.create_tagger(f"{tagger}",
                                    eta_data =df[eta_column].to_numpy(), 
                                    dec_data = df[tagDec_column].to_numpy(), 
                                    B_ID =df[B_ID_var].to_numpy(),
                                    mode = mode,
                                    weight=df["signal_weights"].to_numpy().astype(np.float64), 
                                    tau_ps=df["time"].to_numpy().astype(np.float64),)
                                    #tauerr_ps=df["time_err"].to_numpy().astype(np.float64),) #Need to replace with calibrated time error!!!!
                # Different calibration curves for each tagger, maybe put a if wrt to tagger name
                #tagger_collection[i].set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
                if link == 'logit':
                    if tagger.startswith("OSKaon"):
                        tagger_collection[i].set_calibration(ft.PolynomialCalibration(npar=4, link=ft.link.logit))
                    elif tagger.startswith("OSElectron") and run=="Run2":
                        tagger_collection[i].set_calibration(ft.PolynomialCalibration(npar=3, link=ft.link.logit))
                    else:
                        tagger_collection[i].set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
                    #tagger_collection[i].set_calibration(ft.PolynomialCalibration(npar=2 if not tagger.startswith("OSKaon") else 4, link=ft.link.logit))

                elif link == 'mistag':
                    tagger_collection[i].set_calibration(ft.PolynomialCalibration(npar=2 if not tagger.startswith("OSKaon") or not tagger.startswith("OSElectron") else 4, link=ft.link.mistag))
                #i+=1
            
            #tagger_collection.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
            tagger_collection.calibrate()
            nbins=10
            tagger_collection.plot_calibration_curves(savepath = f'{outputPath}/{run}/{link}', omega_range="minimal", nbins=nbins,)
            os.makedirs(f'{outputPath}/{run}/{link}/enlarged', exist_ok=True)
            tagger_collection.plot_calibration_curves(savepath = f'{outputPath}/{run}/{link}/enlarged', omega_range="minimal", nbins=nbins, x_scale =(lambda x: x**4, lambda x: x**1/4), y_scale =(lambda x: x**4, lambda x: x**1/4))
            ft.save_calibration(tagger_collection, f"{outputPath}/{run}/{link}/single_taggers_calibration.json")
            class_indices = df[B_ID_var].to_numpy()
            #embed()
            class_label_dict = {521: '$B^+$', -521: '$B^-$', 511: '$B^0$', -511: '$\overline{B}^0$', 531: '$B_s^0$', -531: '$\overline{B}_s^0$'}
        # Combine the taggers into one. With "calibrated=False" we would combine the raw single tagger statistics, which is not usually what we want.
            tagger_combination = tagger_collection.combine_taggers(f'{cfg.combinationName}_{run}', calibrated=True)
            if link == 'logit':
                tagger_combination.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
            elif link == 'mistag':
                tagger_combination.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.mistag))
            ## And calibrate this tagger again
            tagger_combination.calibrate()
            print(
                f"Combined tagging efficiency: {tagger_combination.stats.tagging_efficiency(calibrated=True)}")
            print(
                f"Combined tagging power: {tagger_combination.stats.tagging_power(calibrated=True)}")

            #embed()
            ft.plotting.draw_calibration_curve(tagger_combination, savepath=f'{outputPath}/{run}/{link}', nbins=nbins)
            has_ss = any('SS' in t for t in cfg.tagger)
            has_os = any('OS' in t for t in cfg.tagger)
            if has_ss and has_os:
                tagger_combination.name = 'Data OS+SS Run3' if run == 'Run3' else 'Data OS+SS Run2' 
            elif has_ss and not has_os:
                tagger_combination.name = 'Data SS Run3' if run == 'Run3' else 'Data SS Run2'
            elif not has_ss and has_os:
                tagger_combination.name = 'Data OS Run3' if run == 'Run3' else 'Data OS Run2'

            if mode == "Bu":
            # Fucntion chrashing for other modes on data
                tagger_collection.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, 
                            class_label_dict = class_label_dict,file_name = 'single_split_calibration_curves.pdf', 
                            savepath = f'{outputPath}/{run}/{link}', omega_range="minimal", nbins = nbins, x_scale ='linear', y_scale ='linear')
        
                tagger_collection.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, 
                            class_label_dict = class_label_dict,file_name = 'single_split_calibration_curves.pdf', 
                            savepath = f'{outputPath}/{run}/{link}/enlarged', omega_range="minimal", nbins = nbins, x_scale =(lambda x: x**4, lambda x: x**1/4), y_scale =(lambda x: x**4, lambda x: x**1/4))
                ft.plotting.draw_split_calibration_curve(tagger=tagger_combination, nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
                                                    file_name = 'combination_split_calibration_curves.pdf', savepath = f'{outputPath}/{run}/', omega_range="minimal", 
                                                    nbins = nbins, x_scale = 'linear', y_scale = 'linear')#, share_y= True, share_x = True)
            ft.save_calibration(taggers=tagger_combination, title=cfg.combinationName, save_path=f'{outputPath}/{run}/{link}')
            print(f'{run} combination created at {outputPath}')
        
