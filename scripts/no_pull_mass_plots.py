import uproot
import numpy as np
import pandas as pd
import os
from os.path import join
import glob
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("LHCb2")
# import pprint
import argparse
import json

import zfit
from zfit.models.physics import DoubleCB
from zfit.models.physics import GeneralizedCB
from zfit.models.physics import CrystalBall
from zfit.models.basic import Exponential
from zfit.models.functor import SumPDF
from hepstats.splot import compute_sweights


import datetime
import tensorflow as tf
import psutil
from scripts.train_BDT import vars_by_decay
from scripts.train_BDT import KFoldBDT
import pickle


def get_tex_decay(decay , charge = r'\pm'):
    if "Bu2JpsiK" == decay:
        tex_decay = r"$B^{%s} \to J/\psi K^{%s}$" % (charge, charge)
    if "Bd2JpsiKst" == decay:
        tex_decay = r"$B^{0} \to J/\psi K^*$"
    return tex_decay

# def get_mc_names(bdt_features):


def massfit(obs, masses, tex_decay, outname, sim_fit, data_fit, filename, df, compute_weights, generate_figures, obs_name, prefix='', charge = r'\pm'):

    with open(sim_fit) as f:
        _pars = json.load(f)
    alphaL = zfit.Parameter("alphaL", _pars["alphaL"]["value"], -5,      5.0, floating=False)
    nL     = zfit.Parameter("nL",     _pars["nL"]["value"],      0.01, 200.0, floating=False)
    alphaR = zfit.Parameter("alphaR", _pars["alphaR"]["value"], -5,      5.0, floating=False)
    nR     = zfit.Parameter("nR",     _pars["nR"]["value"],      0.01, 200.0, floating=False)







    # Initialise parameters
    # Signal DoubleCB
    with open(data_fit) as f:
        _pars = json.load(f)


    mean     = zfit.Parameter("mean",      _pars['mean']['value'], 5200.0, 5400, floating=False)
    sigmaL   = zfit.Parameter("sigmaL",    _pars['sigmaL']['value'],    0.1,   30, floating=False)
    sigmaR   = zfit.Parameter("sigmaR",    _pars['sigmaR']['value'],    0.1,   30, floating=False)
    g_sigma1 = zfit.Parameter("g_sigma1",  _pars['g_sigma1']['value'],    0.1,   30, floating=False)
    g_sigma2 = zfit.Parameter("g_sigma2",  _pars['g_sigma2']['value'],    0.1,   30, floating=False)
    sig_frac = zfit.Parameter("sig_frac",  _pars['sig_frac']['value'],    0.0,    1, floating=False)
    g_frac1  = zfit.Parameter("g_frac1",   _pars['g_frac1']['value'],    0.0,    1, floating=False)
    # yield_signal = zfit.Parameter("yield_signal", _pars['yield_signal']['value'], 0, len(df), floating=False)
    yield_signal = zfit.Parameter("yield_signal", len(df), 0, len(df), floating=False)
    if not compute_weights: 
        c1 = zfit.Parameter("c1",  _pars['c1']['value'], floating=False)
        c2 = zfit.Parameter("c2",  _pars['c2']['value'], floating=False)
        c3 = zfit.Parameter("c3",  _pars['c3']['value'], floating=False)
        c4 = zfit.Parameter("c4",  _pars['c4']['value'], floating=False)
    else:
        lambd = zfit.Parameter("lambda",  _pars['lambda']['value'], -1, 0, floating=False)
    


    # yield_bkg = zfit.Parameter("yield_bkg", _pars['yield_bkg']['value'], 0, len(df), floating=False)
    yield_bkg = zfit.Parameter("yield_bkg", len(df), 0, len(df), floating=False)

    double_cb = GeneralizedCB(obs=obs, mu=mean, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, nl=nL, alphar=alphaR, nr=nR)
    
    




    gauss1 =zfit.pdf.Gauss(obs=obs, mu=mean, sigma=g_sigma1)
    gauss2 =zfit.pdf.Gauss(obs=obs, mu=mean, sigma=g_sigma2)

    gauss = zfit.pdf.SumPDF([gauss1, gauss2], [g_frac1])

    model_sig = zfit.pdf.SumPDF(
        [double_cb, gauss], [sig_frac])
    model_sig_ext = model_sig.create_extended(yield_signal)
    # model_sig_ext = double_cb.create_extended(yield_signal)

    # Background (Exponential)
    yield_min = len(df) * 0.1 if compute_weights else len(df) * 0.5 #compute_weights also means BDT selection happened


    if not compute_weights: # Pre BDT cuts the background is not exponentially shaped -> chebyshev
        bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1, c2, c3, c4])
    else:
        bkg_model = zfit.pdf.Exponential(obs=obs, lambda_=lambd)

    comb_ext = bkg_model.create_extended(yield_bkg)

    model = zfit.pdf.SumPDF([model_sig_ext, comb_ext])


    # Convert data
    data = zfit.Data.from_numpy(obs=obs, array=masses)
    # Save to JSON file
    os.makedirs(outputdir, exist_ok=True)

    x_plot = np.linspace(mass_range[0], mass_range[1], 1000)
    signal_pdf_eval = model_sig.pdf(x_plot, norm_range=obs)
    sig_yield_val = _pars['yield_signal']['value']



    bins=100
    binwidth = (mass_range[1] - mass_range[0] )/bins
    signal_scaled = sig_yield_val * signal_pdf_eval * binwidth
    # gauss_scaled = gauss_yield_val * gauss_pdf_eval * binwidth
    total_signal_eval = signal_scaled

    background_pdf_eval = bkg_model.pdf(x_plot, norm_range=obs)
    # background_pdf_eval = exp.pdf(x_plot, norm_range=obs)
    bkg_yield_val = _pars['yield_bkg']['value']
    bkg_scaled = bkg_yield_val * background_pdf_eval * binwidth
    total_pdf_eval = signal_scaled + bkg_scaled



    if generate_figures:
        print(f'Generating figures for {tex_decay} fit', flush=True)
        # Plot the results
        fig, (ax1) = plt.subplots(1, 1)

        ax1.plot(x_plot, bkg_scaled, label="Combinatorial", color="green", linestyle = "--", linewidth=2)
        ax1.plot(x_plot, total_pdf_eval, label="Total Fit", color='red',linewidth=3)



        # Scatter plot of data points with errors
        counts, bin_edges = np.histogram(masses, bins=bins, range=mass_range)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  
        errors = np.sqrt(counts)
        lab = "Data"
        ax1.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', label=lab, markersize=3, elinewidth=1.0, capsize=3, capthick=1.0)

        # Compute fit curve

        ax1.plot(x_plot, total_signal_eval, label=tex_decay, color="blue", linestyle = "--", linewidth=2)




        ylabel = f"Candidates$~/~${binwidth}" + r"$~\mathrm{MeV}\!/c^2$"
        ax1.set_ylabel(ylabel)
        ax1.set_yscale('log')
        ax1.legend()




        if "Bu2JpsiK" in cfg.decayType:
            xlabel = r"$ m(J/\psi K^{%s})~[\mathrm{MeV}/c^2]$" % charge
        elif "Bd2JpsiKst" in cfg.decayType:
            xlabel = r"$ m(J/\psi K^{*})~[\mathrm{MeV}/c^2]$"
        else:
            raise ValueError(f"Unknown decay type: {cfg.decayType}")

        ax1.set_xlabel(xlabel)
        ax1.set_xlim(mass_range[0], mass_range[1])

        y_min = np.min(bkg_scaled)
        if y_min <= 2:
            y_min = 2
        ax1.set_ylim(y_min/2, np.max(total_pdf_eval)*2)

        plt.tight_layout()
        save_path = join(outputdir, prefix + filename)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        plt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_files', nargs='+')
    parser.add_argument('--treename', help='TreeName of the input file', type=str, default="BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree")
    parser.add_argument('--obs', help='Observable to fit', type=str, default="B_DTF_PV_Jpsi_MASS")
    parser.add_argument('--range', help='Observable range', nargs="+")
    parser.add_argument('--output', help='Where fit results and plots will be stored', type=str)
    parser.add_argument('--sim_fit', help="Fit results from mc")
    parser.add_argument('--data_fit', help="Fit results from data", type=str)
    parser.add_argument('--sim_files', help="MC files", nargs="+")
    parser.add_argument('--decayType', help='Decay used', type=str)
    parser.add_argument('--seed', help='RNG seed used', type=int)
    parser.add_argument('--cut', help='Cut desired', type=str)
    parser.add_argument('--BDT', help='Path of BDT for event selection') 
    parser.add_argument('--obs_name', help='Name of observable', type=str, default="B_DTF_PV_Jpsi_MASS")
    parser.add_argument('--num_threads', help='Number of threads to use for the fits', type=int, default=1)

    # to do parse background model 

    cfg = parser.parse_args()
    # pprint(cfg)

    # id = ''
    # if cfg.sim_fit:
    #     id =  os.path.basename(cfg.data_files)[:-5] + '_'    
    massname = cfg.obs
    mass_range = (int(cfg.range[0]), int(cfg.range[1]))
    outputdir = join(cfg.output, "data_no_pull_plot")
    os.makedirs(outputdir, exist_ok=True)

    zfit.run.set_n_cpu(n_cpu=cfg.num_threads)


    print(f'Reading files started on {datetime.datetime.now().strftime("%H:%M:%S")}')
    BDT = None
    bdt_features = vars_by_decay[cfg.decayType]

    with open(cfg.BDT, 'rb') as f:
        loaded_data = pickle.load(f)

    BDT = loaded_data['model']        
    cut   = loaded_data['cut']

    vars = ['RUNNUMBER', 'EVENTNUMBER', massname, 'B_ID']
    
    input_files = cfg.data_files
    if isinstance(input_files, str):
        input_files = [input_files]
    # Loop over all files
    dataframes = []





    for i, f in enumerate(input_files):
        print(f"Reading input file {i+1}/{len(input_files)}: {f}")
        print(f'Megabites used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}', flush=True)

        sample_number = int(os.path.basename(f)[10:-14])  # to match weights with the data in add_weights.py

        filenumber = os.path.basename(f)[:-7]


        with uproot.open(f) as _f:
            _df = _f[cfg.treename].arrays(vars+bdt_features, library="pd")
        _df.dropna(inplace=True)
        _df['SAMPLENUMBER'] = sample_number
        _df["event_entry"] = filenumber + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        # _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'SAMPLENUMBER'], inplace=True)

        _df = _df.groupby("event_entry").first()
        _df.reset_index(inplace=True)
        _df['signalness'] = BDT.predict_proba(_df[bdt_features].to_numpy(), _df["event_entry"].values)
        _df.drop(columns=bdt_features, inplace=True)

        dataframes.append(_df)
    #print(f'{pd.concat(singleTagger_dataframes).shape[0]}')
    df_data=pd.concat(dataframes, ignore_index=True)
    del dataframes

    print(f'Reading files ended on {datetime.datetime.now().strftime("%H:%M:%S")}')


    df_data = df_data.query(f'{cfg.obs} < {mass_range[1]} and {cfg.obs} > {mass_range[0]}')

    masses = df_data[massname].values
    obs = zfit.Space("mass", limits=mass_range)


    tex_decay = get_tex_decay(cfg.decayType, )


    data_fit = cfg.data_fit.replace('after_cut', 'before_cut') 
    print(data_fit)

    res_name = "data_res"
    massfit(obs, masses, tex_decay, res_name+'_before_cut', cfg.sim_fit, data_fit, f"fit_res_before_cut.pdf", df_data, 
            compute_weights=False, generate_figures=True, obs_name=cfg.obs_name)


    
    
    pd.set_option('display.max_columns', 15)

    sim_fit_after_cut = cfg.sim_fit.replace('before_cut', 'after_cut')


    df_data = df_data[df_data['signalness'] > cut]
    df_data['BID_signal_weights'] = 0
    df_data['BID_background_weights'] = 0
    for id in df_data['B_ID'].unique():
        print(f'Calculating Sweights for BID={id}')
        charge = "+" if id > 0 else "-"
        tex_decay = get_tex_decay(cfg.decayType, charge=charge)
        data_fit = cfg.data_fit.replace('after_cut', 'after_cut_BID' + str(id))

        df_fit = df_data[df_data['B_ID'] == id]

        
        masses  = df_fit[massname].values
        massfit(obs, masses, tex_decay, res_name+f'_after_cut_BID{id}', sim_fit_after_cut,data_fit, f"fit_after_cut.pdf", df_fit, 
                compute_weights= True, generate_figures= True, obs_name = cfg.obs_name, prefix=f'BID{id}_', charge=charge)
    del df_fit

    masses  = df_data[massname].values
    tex_decay = get_tex_decay(cfg.decayType, )
    massfit(obs, masses, tex_decay, res_name+'_after_cut', sim_fit_after_cut,cfg.data_fit, f"fit_after_cut.pdf", df_data, True, True, cfg.obs_name, )


    print(f'Mass fit script ended on {datetime.datetime.now().strftime("%H:%M:%S")}')
    