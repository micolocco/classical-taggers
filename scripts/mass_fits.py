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


def get_tex_decay(decay):
    if "Bu2JpsiK" == decay:
        tex_decay = r"$B^+ \to J/\psi K^+$"
    if "Bd2JpsiKst" == decay:
        tex_decay = r"$B^{0} \to J/\psi K^*$"
    return tex_decay

# def get_mc_names(bdt_features):


def massfit(obs, masses, tex_decay, outname, simulation, sim_fit, filename, df, compute_weights, generate_figures, obs_name, prefix=''):
    print(f"Starting mass fit for {tex_decay}", flush=True)
    if not simulation: # Fix signal tail shape from MC
        with open(sim_fit) as f:
            _pars = json.load(f)
        alphaL = zfit.Parameter("alphaL", _pars["alphaL"]["value"], -5,      5.0, floating=False)
        nL     = zfit.Parameter("nL",     _pars["nL"]["value"],      0.01, 200.0, floating=False)
        alphaR = zfit.Parameter("alphaR", _pars["alphaR"]["value"], -5,      5.0, floating=False)
        nR     = zfit.Parameter("nR",     _pars["nR"]["value"],      0.01, 200.0, floating=False)


    else:
        alphaL = zfit.Parameter("alphaL",  3.0,  0,      5.0, floating=True)
        nL     = zfit.Parameter("nL",      1.7,  0.01, 200.0, floating=True)
        alphaR = zfit.Parameter("alphaR",  3.0,  0.0,    5.0, floating=True)
        nR     = zfit.Parameter("nR",      1.7,  0.01, 200.0, floating=True)
        






    # Initialise parameters
    # Signal DoubleCB
    mean     = zfit.Parameter("mean",   5300.0, 5200.0, 5400)
    sigmaL   = zfit.Parameter("sigmaL",   12.0,    0.1,   30)
    sigmaR   = zfit.Parameter("sigmaR",   12.0,    0.1,   30)
    g_sigma1 = zfit.Parameter("g_sigma1",  6.0,    0.1,   30)
    g_sigma2 = zfit.Parameter("g_sigma2", 10.0,    0.1,   30)
    sig_frac = zfit.Parameter("sig_frac",  0.5,    0.0,    1)
    g_frac1  = zfit.Parameter("g_frac1",   0.1,    0.0,    1)

    yield_signal = zfit.Parameter("yield_signal", len(df), 0, len(df))
    
    double_cb = GeneralizedCB(obs=obs, mu=mean, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, nl=nL, alphar=alphaR, nr=nR)
    
    




    gauss1 =zfit.pdf.Gauss(obs=obs, mu=mean, sigma=g_sigma1)
    gauss2 =zfit.pdf.Gauss(obs=obs, mu=mean, sigma=g_sigma2)
    g_frac1 = zfit.Parameter("g_frac1",   0.1, 0, 1, floating=True)

    gauss = zfit.pdf.SumPDF([gauss1, gauss2], [g_frac1])

    model_sig = zfit.pdf.SumPDF(
        [double_cb, gauss], [sig_frac])
    model_sig_ext = model_sig.create_extended(yield_signal)
    # model_sig_ext = double_cb.create_extended(yield_signal)

    if not simulation:
        # Background (Exponential)
        yield_min = len(df) * 0.1 if compute_weights else len(df) * 0.5 #compute_weights also means BDT selection happened
        yield_bkg = zfit.Parameter("yield_bkg", yield_min, 0, len(df))


        if not compute_weights: # Pre BDT cuts the background is not exponentially shaped -> chebyshev
            c1 = zfit.Parameter("c1",  0.1)
            c2 = zfit.Parameter("c2", -0.2)
            c3 = zfit.Parameter("c3",  0.0)
            c4 = zfit.Parameter("c4",  0.0)
            bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1, c2, c3, c4])
        else:
            lambd = zfit.Parameter("lambda", -0.01, -1, 0) 
            bkg_model = zfit.pdf.Exponential(obs=obs, lambda_=lambd)

        comb_ext = bkg_model.create_extended(yield_bkg)

        model = zfit.pdf.SumPDF([model_sig_ext, comb_ext])

    else:
        model = model_sig_ext


    # Convert data
    data = zfit.Data.from_numpy(obs=obs, array=masses)
    # Perform fit
    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit(gradient=False, tol= 1e-4) #Maybe look into using tol, or strategy might work better??
    result = minimizer.minimize(nll)

    print(f"Initial fit converged: {result.converged} with edm: {result.edm}", flush=True)
    print(f"Initial fit is valid: {result.valid}", flush=True)

    if not result.valid or result.edm > 1e-4:  # If the fit is not valid or edm is too high, try again
        print(f"Initial fit is not valid or edm is to high, trying again starting with the values from result", flush=True)
        result = minimizer.minimize(nll, init=result)
    
    result.hesse()
    cov_matrix = result.covariance()
    params = result.params
    print(result, flush=True)
    print(f'Result message: {result.message}', flush=True)
    
    
    # Save the results
    params = result.params  # Get the fitted parameters

    # Extract parameter names and values
    fit_results = {param.name: {"value": info["value"], "error": info["hesse"]["error"]} for param, info in params.items()}

    # Save to JSON file
    os.makedirs(outputdir, exist_ok=True)

    with open(join(outputdir, f'{outname}.json'), "w") as f:
        json.dump(fit_results, f, indent=4)

    x_plot = np.linspace(mass_range[0], mass_range[1], 1000)
    signal_pdf_eval = model_sig.pdf(x_plot, norm_range=obs)
    sig_yield_val = params[yield_signal]['value']



    bins=100
    binwidth = (mass_range[1] - mass_range[0] )/bins
    signal_scaled = sig_yield_val * signal_pdf_eval * binwidth
    # gauss_scaled = gauss_yield_val * gauss_pdf_eval * binwidth
    total_signal_eval = signal_scaled

    if not simulation:
        background_pdf_eval = bkg_model.pdf(x_plot, norm_range=obs)
        # background_pdf_eval = exp.pdf(x_plot, norm_range=obs)
        bkg_yield_val = params[yield_bkg]['value']
        bkg_scaled = bkg_yield_val * background_pdf_eval * binwidth
        total_pdf_eval = signal_scaled + bkg_scaled
    


    if generate_figures:
        print(f'Generating figures for {tex_decay} fit', flush=True)
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

        if not simulation:
            ax1.plot(x_plot, bkg_scaled, label="Combinatorial", color="green", linestyle = "--", linewidth=2)
            ax1.plot(x_plot, total_pdf_eval, label="Total Fit", color='red',linewidth=3)



        # Scatter plot of data points with errors
        counts, bin_edges = np.histogram(masses, bins=bins, range=mass_range)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  
        errors = np.sqrt(counts)
        lab = "Data" if not simulation else "MC Data"
        ax1.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', label=lab, markersize=1, elinewidth=1.0)

        # Compute fit curve

        ax1.plot(x_plot, total_signal_eval, label=tex_decay, color="blue", linestyle = "--", linewidth=2)




        ylabel = f"Events$~/~${binwidth}" + r"$[~\mathrm{MeV}/c^2]$"
        ax1.set_ylabel(ylabel)
        ax1.set_yscale('log')
        ax1.legend()


        # Compute residuals
        if simulation:
            total_pdf_eval = signal_scaled
        else:
            total_pdf_eval = signal_scaled + bkg_scaled
        total_fit_at_bin_centers = np.interp(bin_centers, x_plot, total_pdf_eval)
        residuals = (counts - total_fit_at_bin_centers) / errors
        print(f"Residuals: {residuals}", flush=True)
        ax2.axhline(0, color='black', linestyle='dashed')
        ax2.axhline(2, color='red', linestyle='dotted')
        ax2.axhline(-2, color='red', linestyle='dotted')
        ax2.hist(bin_centers, weights=residuals, bins=bins, histtype='step', linestyle='-', linewidth=1.5, facecolor='gray', alpha=0.5, fill = True, color = 'gray')
        ax2.set_ylim(-5, 5)        
        ax2.set_ylabel("Pull")

        if "Bu2JpsiK" in cfg.decayType:
            xlabel = r"$ m(J/ψK^{\pm})~[\mathrm{MeV}/c^2]$"
        elif "Bd2JpsiKst" in cfg.decayType:
            xlabel = r"$ m(J/ψK^{*})~[\mathrm{MeV}/c^2]$"
        else:
            raise ValueError(f"Unknown decay type: {cfg.decayType}")

        ax2.set_xlabel(xlabel)
        ax1.set_xlim(mass_range[0], mass_range[1])

        y_min = np.min(bkg_scaled) if not simulation else np.min(signal_scaled)
        if y_min <= 2:
            y_min = 2
        ax1.set_ylim(y_min/2, np.max(counts)*2)

        plt.tight_layout()
        plt.savefig(join(outputdir, prefix + filename))
        plt.close()

    if compute_weights:
        print('computing SWeights', flush=True)
        weights = compute_sweights(model, masses)



        df[f"{prefix}signal_weights"] = weights[yield_signal] 
        df[f"{prefix}background_weights"] = weights[yield_bkg] 
        del weights

        if generate_figures:
            plt.plot(masses, df[f"{prefix}signal_weights"], marker=".", linestyle="None", color="red", markersize=0.1, label="signal")
            plt.plot(masses, df[f"{prefix}background_weights"], marker=".", linestyle="None", color="green", markersize=0.1, label="background weights")
            plt.plot(masses, df[f"{prefix}background_weights"] + df[f"{prefix}signal_weights"], marker=".", linestyle="None", color="black", markersize=0.1, label="Sum of three")
            plt.xlabel("m($B^{+})~[MeV]/c^{2}$")
            plt.ylabel("weights")
            plt.legend()
            plt.savefig(join(outputdir, prefix + f"validate_sweights.png"))
            plt.close()

        #Calculate and save the pdf_ratio
        signal = model_sig_ext.pdf( df[obs_name], obs) * sig_yield_val
        bkg = comb_ext.pdf( df[obs_name], obs) * bkg_yield_val
        df["pdf_ratio"] = signal/bkg



    print(f"Fit ended with status: {result.status}", flush=True)



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
    parser.add_argument('--simulation', action="store_true")
    parser.add_argument('--sim_fit', help="Fit results from mc")
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
    outputdir = join(cfg.output, "mc_fit") if cfg.simulation else join(cfg.output, "data_fit")
    os.makedirs(outputdir, exist_ok=True)

    zfit.run.set_n_cpu(n_cpu=cfg.num_threads)


    print(f'Reading files started on {datetime.datetime.now().strftime("%H:%M:%S")}')
    BDT = None
    bdt_features = vars_by_decay[cfg.decayType]

    with open(cfg.BDT, 'rb') as f:
        loaded_data = pickle.load(f)

    BDT = loaded_data['model']        
    cut   = loaded_data['cut']

    if not cfg.simulation:
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

    else:
        df_data = pd.DataFrame()
        input_files = cfg.sim_files

        # bdt_features = get_mc_names(bdt_features)

        for i, file in enumerate(input_files):
            print(f"Reading input file {i+1}/{len(input_files)}: {file}", flush=True)
            filenumber = os.path.basename(file)[:-3] + 'mc'

            with uproot.open(file) as f:
                _df = f[cfg.treename].arrays([massname, 'RUNNUMBER', 'EVENTNUMBER', 'B_BKGCAT']+ bdt_features, library="pd")
            _df.dropna(inplace=True)
            _df = _df.query("B_BKGCAT == 0")  
            
            _df["event_entry"] = filenumber + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'B_BKGCAT'], inplace=True)

            _df = _df.groupby("event_entry").first()
            _df.reset_index(inplace=True)
            _df['signalness'] = BDT.predict_proba(_df[bdt_features].to_numpy(), _df["event_entry"].values)
            _df.drop(columns=bdt_features, inplace=True)


            df_data = pd.concat([df_data, _df], ignore_index = True)
    print(f'Reading files ended on {datetime.datetime.now().strftime("%H:%M:%S")}')


    df_data = df_data.query(f'{cfg.obs} < {mass_range[1]} and {cfg.obs} > {mass_range[0]}')

    masses = df_data[massname].values
    obs = zfit.Space("mass", limits=mass_range)


    tex_decay = get_tex_decay(cfg.decayType)




    res_name = "mc_res" if cfg.simulation else "data_res"
    massfit(obs, masses, tex_decay, res_name+'_before_cut', cfg.simulation, cfg.sim_fit, f"fit_res_before_cut.png", df_data, 
            compute_weights=False, generate_figures=True, obs_name=cfg.obs_name)


    if cfg.simulation:
        print(f'Running mass fit for simulation with cut {cut}', flush=True)
        df_data = df_data.query(f'signalness > {cut}')
        masses  = df_data[massname].values
        massfit(obs, masses, tex_decay, res_name+'_after_cut', cfg.simulation, cfg.sim_fit, f"fit_res_after_cut.png", df_data, False, True, cfg.obs_name)
    else:
        pd.set_option('display.max_columns', 15)

        sim_fit_after_cut = cfg.sim_fit.replace('before_cut', 'after_cut')


        df_data = df_data[df_data['signalness'] > cut]
        df_data['BID_signal_weights'] = 0
        df_data['BID_background_weights'] = 0
        for id in df_data['B_ID'].unique():
            print(f'Calculating Sweights for BID={id}')
            df_fit = df_data[df_data['B_ID'] == id]

            
            masses  = df_fit[massname].values
            massfit(obs, masses, tex_decay, res_name+f'_after_cut_BID{id}', cfg.simulation, sim_fit_after_cut, f"fit_after_cut.png", df_fit, 
                    compute_weights= True, generate_figures= True, obs_name = cfg.obs_name, prefix=f'BID{id}_', )
            
            # print(df_data.head(10))

            # print(df_data.loc[df_data['B_ID'] == id])
            # print(df_fit['BID_signal_weights'].values)
            df_data.loc[df_data['B_ID'] == id, 'BID_signal_weights'] = df_fit[f'BID{id}_signal_weights'].values
            df_data.loc[df_data['B_ID'] == id, 'BID_background_weights'] = df_fit[f'BID{id}_background_weights'].values

        print(f'Calculating total Sweights')
        masses  = df_data[massname].values
        massfit(obs, masses, tex_decay, res_name+'_after_cut', cfg.simulation, sim_fit_after_cut, f"fit_after_cut.png", df_data, True, True, cfg.obs_name, )

        df_data.reset_index(inplace=True)
        df_data.drop(columns=['event_entry'], inplace = True)

        #Save weighted dataframe to disk
        tree_dict = {col: np.array(df_data[col]) for col in df_data.columns if col != 'event_entry'}
        print(tree_dict)

        for col in tree_dict:
            print(f'{col}: {tree_dict[col][0]} ({type(tree_dict[col][0])})')

        del df_data
        print('Writing file to disk')


        with uproot.recreate(join(outputdir, 'weights.root')) as f:
            f['DecayTree'] = tree_dict


    print(f'Mass fit script ended on {datetime.datetime.now().strftime("%H:%M:%S")}')
    