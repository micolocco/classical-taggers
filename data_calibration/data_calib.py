import lhcb_ftcalib as ft
import os
import argparse
from pprint import pprint
import uproot
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine the taggers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input_file', help='Folder where the files with the tagging decision are saved', default='/ceph/users/molocco/Data/withUT_MC_2024/4_tagged')
    parser.add_argument('--tagger', help='List of taggers/single tagger', nargs='+', required=True)
    parser.add_argument('--decayType', help='Decay used for the calibration', type=str)
    parser.add_argument('--outputPath', help='Name of the output dir', type=str, default='/ceph/users/molocco/Data/savedModels/withUT_MC_2024/')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--cut', help='Cut desired', type=str, required=True)
    parser.add_argument('--features', help='Input features used for NN training',) 
    parser.add_argument('--run2', help='If Run2 tagger combination must be computed as well',  action='store_true') # action='store_true' means args.run2 will be set to True if the --Run2 argument is provided on the command line.
    parser.add_argument('--combinationName', help='Name used for the output combination', type=str)

    
    cfg = parser.parse_args()
    pprint(cfg)

    run = "Run2" if "Run2" in cfg.tagger[0] else "Run3"
    outputPath =f'{cfg.outputPath}/{cfg.decayType}/combinations/{run}'
    os.makedirs(outputPath, exist_ok=True)

    B_ID_var = "B_ID"
    with uproot.open(cfg.input_file) as f:
        df = f['DecayTree'].arrays(library="pd")
    taggers = ft.TaggerCollection()
    # run = "Run3"
    i=0
    for tagger in cfg.tagger:
        if run == "Run3":
            taggers.create_tagger(f"{tagger}", eta_data =df[f'{tagger}_Eta'].tolist(), dec_data = df[f'{tagger}_TagDec'].tolist(), B_ID =df[B_ID_var].tolist(), mode = 'Bu', weight=df["signal_weights"].to_numpy().astype(np.float64))
        else:
            taggers.create_tagger(f"{tagger}", eta_data =df[f'{tagger}_Omega'].tolist(), dec_data = df[f'{tagger}_Dec'].tolist(), B_ID =df[B_ID_var].tolist(), mode = 'Bu', weight=df["signal_weights"].to_numpy().astype(np.float64))
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
