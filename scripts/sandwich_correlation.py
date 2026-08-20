
import uproot 
import pandas as pd
import os
import numpy as np
import argparse
from pprint import pprint
from rich.console import Console
from rich.table import Table
from rich import box
from io import StringIO
import matplotlib.pyplot as plt

import lhcb_ftcalib as ft
from xarray import corr, cov


def load_dataframes(root_paths, labels, base_vars, taggers):
    df = None

    for path, label in zip(root_paths, labels):
        if 'V0' in label:
            vars = ['B_Run2_' + tagger + '_Omega' for tagger in taggers] + ['B_Run2_' + tagger + '_Dec' for tagger in taggers]
        else:
            vars = [tagger + '_Eta' for tagger in taggers] + [tagger + '_TagDec' for tagger in taggers]


        with uproot.open(path) as f:
            _df = f[cfg.tree].arrays(base_vars + vars, library="pd")
            _df["label"] = label

        if 'V0' in label:
            rename_dict = {f'B_Run2_{tagger}_Omega': f'{tagger}_Eta' for tagger in taggers}
            rename_dict.update({f'B_Run2_{tagger}_Dec': f'{tagger}_TagDec' for tagger in taggers})
            _df.rename(columns=rename_dict, inplace=True)

        if 'index' in _df.columns:
            _df.drop(columns=['index'], inplace=True)

        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df], ignore_index=True).reset_index(drop=True)
    return df


def hessian(tagger, squared_weight = False):
    """ Likelihood hessian """

    func = tagger.func
    # params = np.asarray([tagger.minimizer.values[n] for n in tagger.minimizer.parameters], dtype=np.float64)    
    params = np.asarray(tagger.minimizer.values, dtype=np.float64)

    # params = tagger.stats.params.params_delta
    data = tagger.stats._tagdata

    dilution = 0.5 * (1.0 - np.abs(data.Amix))

    dim      = func.total_npar

    # Compute likelihood terms
    if func.delta_split:
        omega_given = func.eval(params, data.eta,      data.prod_flav)
        omega_oscil = func.eval(params, data.eta, -1 * data.prod_flav)
    else:
        omega_given = func.eval(params, data.eta,      data.prod_flav)
        omega_oscil = omega_given


    Pi = (1.0 - omega_given) * (1.0 - dilution) + omega_oscil * dilution

    hesse = np.zeros((dim, dim))

    domega_given = func.gradient(params, data.eta,      data.prod_flav)  # Dispatches to gradient_split or gradient_averaged
    domega_oscil = func.gradient(params, data.eta, -1 * data.prod_flav)



    for i in range(dim):
        dPi = -domega_given[i] * (1.0 - dilution) + domega_oscil[i] * dilution

        for j in range(dim):
            dPj = -domega_given[j] * (1.0 - dilution) + domega_oscil[j] * dilution

            vals = np.zeros_like(data.eta)
            vals[data.correct_tags] = dPi[data.correct_tags] * dPj[data.correct_tags] / Pi[data.correct_tags]**2
            vals[data.wrong_tags]   = dPi[data.wrong_tags] * dPj[data.wrong_tags] / (1.0 - Pi[data.wrong_tags])**2

            weights = data.weight


            if squared_weight:
                hesse[i][j] = np.sum(data.weight**2 * vals)
            else:
                hesse[i][j] = np.sum(data.weight * vals)
    return hesse


def corrected_1Tagger_covariance(tagger):
    A = hessian(tagger, squared_weight = False)

    B = hessian(tagger, squared_weight = True)

    Ainv = np.linalg.inv(A)

    cov = Ainv @ B @ Ainv

    corr = np.zeros_like(cov)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            corr[i][j] = cov[i][j] / np.sqrt(cov[i][i] * cov[j][j])

    


    print("\n\n\n\n\n Tagger starts", flush = True)
    print("\n\n A starts", flush = True)
    print(A)
    print("\n\n B starts", flush = True)
    print(B)
    print("\n\n Ainv starts", flush = True)
    print(Ainv)
    print("\n\n cov starts", flush = True)
    print(cov)
    print("\n\n corr starts", flush = True)
    print(corr)

    print("\n\n Dilution squared starts", flush = True)
    print(tagger.stats.dilution_squared(calibrated = True))
    print(tagger.stats.dilution_squared(calibrated = False))
    print("Tagger ends \n\n\n\n\n ", flush = True)


def get_score_vector(tagger):

    func = tagger.func
    params = np.asarray(tagger.minimizer.values, dtype=np.float64)
    data = tagger.stats._full_data

    dilution = 0.5 * (1.0 - np.abs(data.Amix))
    dim      = func.total_npar

    # Compute likelihood terms
    if func.delta_split:
        omega_given = func.eval(params, data.eta,      data.prod_flav)
        omega_oscil = func.eval(params, data.eta, -1 * data.prod_flav)
    else:
        omega_given = func.eval(params, data.eta,      data.prod_flav)
        omega_oscil = omega_given
    Pi = (1.0 - omega_given) * (1.0 - dilution) + omega_oscil * dilution

    domega_given = func.gradient(params, data.eta,      data.prod_flav)  # Dispatches to gradient_split or gradient_averaged
    domega_oscil = func.gradient(params, data.eta, -1 * data.prod_flav)

    correct_tags = data.dec == data.prod_flav
    wrong_tags   = data.dec != data.prod_flav

    score_vector = np.zeros((dim, len(data.eta)))

    for i in range(dim):
        dPi = -domega_given[i] * (1.0 - dilution) + domega_oscil[i] * dilution
        
        score_vector[i][correct_tags] = dPi[correct_tags] / Pi[correct_tags]
        score_vector[i][wrong_tags]   = dPi[wrong_tags] / (1.0 - Pi[wrong_tags])


    score_vector[:,~data.tagged_sel] = 0.0

    return score_vector

def get_sandwich_estimator(tagger1, tagger2):
    score_vector1= get_score_vector(tagger1)
    score_vector2= get_score_vector(tagger2)

    dim1 = score_vector1.shape[0]
    dim2 = score_vector2.shape[0]

    total_score = np.concatenate((score_vector1, score_vector2), axis=0)

    A = np.zeros((dim1 + dim2, dim1 + dim2))

    weights = tagger1.stats._full_data.weight


    for i in range(dim1 + dim2):
        for j in range(dim1 + dim2):
            A[i][j] = np.sum(total_score[i] * total_score[j]*weights)


    A_corrected = A.copy()
    A_corrected[dim1:dim1+dim2, 0:dim1] = 0.0
    A_corrected[0:dim1, dim1:dim1+dim2] = 0.0


    Ainv = np.linalg.inv(A_corrected)

    B = np.zeros((dim1 + dim2, dim1 + dim2))

    for i in range(dim1 + dim2):
        for j in range(dim1 + dim2):
            B[i][j] = np.sum(total_score[i] * total_score[j]*weights**2)


    cov_flav_base = Ainv @ B @ Ainv


    corr = np.zeros_like(cov_flav_base)
    for i in range(cov_flav_base.shape[0]):
        for j in range(cov_flav_base.shape[1]):
            corr[i][j] = cov_flav_base[i][j] / np.sqrt(cov_flav_base[i][i] * cov_flav_base[j][j])

    




    print("\n\n\n\n\n 1Tagger starts", flush = True)
    print("\n\n A starts", flush = True)
    print(A)
    print("\n\n A_corrected starts", flush = True)
    print(A_corrected)
    print("\n\n B starts", flush = True)
    print(B)
    print("\n\n Ainv starts", flush = True)
    print(Ainv)
    print("\n\n cov_flav_base starts", flush = True)
    print(cov_flav_base)
    print("\n\n corr starts", flush = True)
    print(corr)
    print("1Tagger ends \n\n\n\n\n ", flush = True)


    delOmegadelp_tagger1 = tagger1.stats.func_ref.gradient_averaged(tagger1.stats.params.params_average, tagger1.stats._tagdata.eta)
    delOmegadelp_tagger2 = tagger2.stats.func_ref.gradient_averaged(tagger2.stats.params.params_average, tagger2.stats._tagdata.eta)



    tagger1_sum_weights = tagger1.stats.cal_Nwts
    tagger2_sum_weights = tagger2.stats.cal_Nwts
    npar1 = tagger1.stats.params.npar
    npar2 = tagger2.stats.params.npar

    delDsqdelp_tagger1 = np.zeros(npar1 + npar2)
    delDsqdelp_tagger2 = np.zeros(npar1 + npar2)

    D_tagger1 = np.array(1 - 2*tagger1.stats._tagdata.omega)
    D_tagger2 = np.array(1 - 2*tagger2.stats._tagdata.omega)


    delDsqdelp_tagger1[:npar1]                = -4 * np.sum(tagger1.stats._tagdata.weight.to_numpy() * D_tagger1 * delOmegadelp_tagger1, axis=1)/tagger1_sum_weights
    delDsqdelp_tagger2[npar1:npar1+npar2] = -4 * np.sum(tagger2.stats._tagdata.weight.to_numpy() * D_tagger2 * delOmegadelp_tagger2, axis=1)/tagger2_sum_weights





    tagger1_base = np.block([[tagger1.stats.params._flavour2delta, np.zeros((dim1, dim2))], [np.zeros((dim2, dim1)), np.eye(dim2)]])

    tagger2_base = np.block([[np.eye(dim1), np.zeros((dim1, dim2))], [np.zeros((dim2, dim1)), tagger2.stats.params._flavour2delta]])

    cov_delta_base =  tagger2_base @ tagger1_base @ cov_flav_base @ tagger1_base.T @ tagger2_base.T

    print("\n\n cov_delta_base starts", flush = True)
    print(cov_delta_base)


    cov_ave_base = np.block([
        [cov_delta_base[   0:npar1,      0:npar1], cov_delta_base[   0:npar1,      dim1:dim1+npar2]], 
        [cov_delta_base[dim1:dim1+npar2, 0:npar1], cov_delta_base[dim1:dim1+npar2, dim1:dim1+npar2]]
    ])



    print("\n\n cov_ave_base starts", flush = True)
    print(cov_ave_base)


    delDsqdelp_stacked = np.stack((delDsqdelp_tagger1, delDsqdelp_tagger2), axis=0)

    cov_Dsq = delDsqdelp_stacked @ cov_ave_base @ delDsqdelp_stacked.T




    corr_Dsq = np.zeros_like(cov_Dsq)
    for i in range(cov_Dsq.shape[0]):
        for j in range(cov_Dsq.shape[1]):
            corr_Dsq[i][j] = cov_Dsq[i][j] / np.sqrt(cov_Dsq[i][i] * cov_Dsq[j][j])

    


    print("\n\n cov D^2 starts", flush = True)
    print(cov_Dsq)
    print('Uncertainty of D^2 for tagger 1:', np.sqrt(cov_Dsq[0][0]))
    print('Uncertainty of D^2 for tagger 2:', np.sqrt(cov_Dsq[1][1]))
    print("\n\n corr D^2 starts", flush = True)
    print(corr_Dsq)
    print("1Tagger ends \n\n\n\n\n ", flush = True)

    tagger1_eff = tagger1.stats.tagging_efficiency(calibrated = True)[0]
    tagger2_eff = tagger2.stats.tagging_efficiency(calibrated = True)[0]


    # jacobi_tagpower = np.array([[tagger1_eff, np.sqrt(tagger1_eff * tagger2_eff)], [np.sqrt(tagger1_eff * tagger2_eff), tagger2_eff]])
    jacobi_tagpower = np.array([[tagger1_eff, 0], [0, tagger2_eff]])

    cov_tagpower = jacobi_tagpower @ cov_Dsq @ jacobi_tagpower.T

    corr_tagpower = np.zeros_like(cov_tagpower)
    for i in range(cov_tagpower.shape[0]):
        for j in range(cov_tagpower.shape[1]):
            corr_tagpower[i][j] = cov_tagpower[i][j] / np.sqrt(cov_tagpower[i][i] * cov_tagpower[j][j])

    print("\n\n cov tagging power starts", flush = True)
    print(cov_tagpower)
    print('Uncertainty of tagging power for tagger 1:', np.sqrt(cov_tagpower[0][0]))
    print('Uncertainty of tagging power for tagger 2:', np.sqrt(cov_tagpower[1][1]))
    print("\n\n corr tagging power starts", flush = True)
    print(corr_tagpower)
    







'''
Comparison of Run3v0 and Run3v1


TEMP
python scripts/sandwich_correlation.py --labels V0 V1 --combined_root_files /ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root /ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/_combined_tagged.root --calibration_files /ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json /ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json --taggers SSPion SSProton OSKaon OSMuon OSElectron



python scripts/sandwich_correlation.py --labels V0 V1 --combined_root_files /ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root /ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root --calibration_files /ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json /ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json --taggers SSPion SSProton OSKaon OSMuon OSElectron


'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to estimate correlation between tagging power uncertainties using the a sandwich estimator.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--labels',              type=str, nargs='+', required=True, help='Labels for the different combined ROOT files.')
    parser.add_argument('--combined_root_files', type=str, nargs='+', required=True, help='Paths to the combined ROOT files for each label. Must be in the same order as the labels.')
    parser.add_argument('--calibration_files',   type=str, nargs='+', required=True, help='Paths to the calibration files for each label. Must be in the same order as the labels.')
    parser.add_argument('--taggers',             type=str, nargs='+', required=True, help='Names of the taggers to use for the correlation calculations.')
    parser.add_argument('--mode',                type=str,                           help='Mode of the decay. Used to determine the correct calibration file to use.', default='Bd',)
    parser.add_argument('--tree',                type=str,                           help='Name of the tree in the ROOT files to read.', default='DecayTree;1',)
    parser.add_argument('--outpath',             type=str,                           help='Path to where the output should be saved',    default='/ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff')
    cfg = parser.parse_args()

    np.set_printoptions(linewidth=10_000)


    event_vars = ['signal_weights', 'B_TAU', 'B_ID'] 

    df = load_dataframes(cfg.combined_root_files, cfg.labels, event_vars, cfg.taggers)


    combined_taggers = []
    for label, calib_file in zip(cfg.labels, cfg.calibration_files):

        df_label = df[df['label'] == label]#[:1_000]
        tagger_collection = ft.TargetTaggerCollection()
        for tagger in cfg.taggers:
            tagger_obj = ft.TargetTagger(tagger,
                                         eta_data  =df_label[f'{tagger}_Eta'].to_numpy(), 
                                         dec_data  =df_label[f'{tagger}_TagDec'].to_numpy(), 
                                         B_ID      =df_label['B_ID'].to_numpy(), 
                                         tau_ps    =df_label['B_TAU'].to_numpy(), 
                                         weight    =df_label['signal_weights'].to_numpy(),
                                         mode      =cfg.mode, 
                                        #  tauerr_ps =tau_ps_err,
                                        )
            # tagger_obj.load(calib_file, tagger_name=tagger)
            # tagger_obj.apply()
            tagger_collection.add_taggers(tagger_obj)

        tagger_collection.load_calibrations(calib_file)
        tagger_collection.apply()

        col_tagger = tagger_collection.combine_taggers('combined', calibrated = True)

        col_tagger.load(calib_file, tagger_name='_'.join(cfg.taggers))
        col_tagger.apply()


        print(f"minimizer values: {col_tagger.minimizer.values}", flush = True)

        combined_taggers.append(col_tagger)





    corrected_1Tagger_covariance(combined_taggers[0])
    corrected_1Tagger_covariance(combined_taggers[1])
    get_sandwich_estimator(combined_taggers[0], combined_taggers[1])

    
    
    






    