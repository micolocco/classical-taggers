import lhcb_ftcalib as ft
import os
import argparse
from pprint import pprint
import uproot
import numpy as np
from IPython import embed

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
    parser.add_argument('--output_folder', help='Name of the output dir', type=str,default='block1_data_combinations')
    parser.add_argument('--time-unit', type=str, default="c_ps",
                        help='Unit of the time branches')
    parser.add_argument('--decay-time-branches', type=str, default=["B_DTF_PV_CTAU"], nargs="+", #"B_DTF_PV_CTAUERR"
                        help='Branches names of the decay-time variables (first decay time, second decay-time error).') # Just using decay time for now

    parser.add_argument('--time-unit', type=str, default="c_ps",
                        help='Unit of the time branches')
    parser.add_argument('--decay-time-branches', type=str, default=["B_DTF_PV_CTAU"], nargs="+",
                        help='Branche names of the decay-time variables (first decay time, second decay-time error).') # Just using decay time for now

    
    cfg = parser.parse_args()
    pprint(cfg)

    path =f'{cfg.input_path}/{cfg.decayType}/{cfg.cut}'
    outputPath = f'{path}/{cfg.output_folder}'
    os.makedirs(outputPath, exist_ok=True)

    B_ID_var = "B_ID"
    input_file = f'{path}/data_fit/sweights.root'
    with uproot.open(input_file) as f:
        df = f['DecayTree'].arrays(library="pd")
    
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

    '''
    # Sara's code
    taggers = ft.TaggerCollection()
    i=0
    run = "Run2" if "Run2" in cfg.tagger[0] else "Run3"
    for tagger in cfg.tagger:
        if run == "Run3":
            taggers.create_tagger(f"{tagger}",
                                  eta_data =df[f'{tagger}_Eta'].tolist(),
                                  dec_data = df[f'{tagger}_TagDec'].tolist(),
                                  B_ID =df[B_ID_var].tolist(),
                                  mode = mode,
                                  weight=df["signal_weights"].to_numpy().astype(np.float64), 
                                  tau_ps=df["time"].to_numpy().astype(np.float64),)
        else:
            taggers.create_tagger(f"{tagger}",
                                  eta_data =df[f'{tagger}_Omega'].tolist(),
                                  dec_data = df[f'{tagger}_Dec'].tolist(),
                                  B_ID =df[B_ID_var].tolist(),
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
        os.makedirs(f'{outputPath}/{run}', exist_ok=True)
        taggers = ft.TaggerCollection()
        for i, tagger in enumerate(cfg.tagger):
            if run == "Run2":
                eta_column = f'B_{run}_{tagger}_Omega'
                tagDec_column = f'B_{run}_{tagger}_Dec' 
                print(eta_column)   
            else:
                eta_column = f'{tagger}_Eta'
                tagDec_column = f'{tagger}_TagDec'
            taggers.create_tagger(f"{tagger}",
                                eta_data =df[eta_column].tolist(), 
                                dec_data = df[tagDec_column].tolist(), 
                                B_ID =df[B_ID_var].tolist(),
                                mode = mode,
                                weight=df["signal_weights"].to_numpy().astype(np.float64), 
                                tau_ps=df["time"].to_numpy().astype(np.float64),)
                                #tauerr_ps=df["time_err"].to_numpy().astype(np.float64),) #Need to replace with calibrated time error!!!!
            # Different calibration curves for each tagger, maybe put a if wrt to tagger name
            '''
            if tagger=='OSKaon':
                taggers[i].set_calibration(ft.PolynomialCalibration(npar=3, link=ft.link.logit))
            #    taggers[i].set_calibration(ft.PolynomialCalibration(npar=3, link=ft.link.logit))
            else:
                taggers[i].set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
            '''
            #i+=1
        taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
        taggers.calibrate()
        # Combine the taggers into one. With "calibrated=False" we would combine the raw single tagger statistics, which is not usually what we want.
        tagger_combination = taggers.combine_taggers(f'{cfg.combinationName}_{run}', calibrated=True)
        tagger_combination.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
        ## And calibrate this tagger again
        tagger_combination.calibrate()
        taggers.plot_calibration_curves(savepath = f'{outputPath}/{run}', omega_range="minimal", nbins=10)
        ft.plotting.draw_calibration_curve(tagger_combination, savepath=f'{outputPath}/{run}', nbins=20)
        ft.save_calibration(taggers=tagger_combination, title=cfg.combinationName, save_path=f'{outputPath}/{run}')
        print(f'{run} combination created at {outputPath}')
    
