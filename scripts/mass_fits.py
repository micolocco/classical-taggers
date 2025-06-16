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
from zfit.models.basic import Exponential
from zfit.models.functor import SumPDF
from hepstats.splot import compute_sweights


import tensorflow as tf
import psutil
from scripts.train_BDT import vars_by_decay
import pickle



# from scripts.preSelections import run2_taggers_variables
run2_taggers_variables = [
        'B_Run2_SSPion_Dec',
        'B_Run2_SSPion_Omega',
        'B_Run2_SSPion_MVA',
        'B_Run2_SSKaon_Dec',
        'B_Run2_SSKaon_Omega',
        'B_Run2_SSKaon_MVA',
        'B_Run2_SSProton_Dec',
        'B_Run2_SSProton_Omega',
        'B_Run2_SSProton_MVA',
        'B_Run2_OSKaon_Dec',
        'B_Run2_OSKaon_Omega',
        'B_Run2_OSKaon_MVA',
        'B_Run2_OSElectron_Dec',
        'B_Run2_OSElectron_Omega',
        'B_Run2_OSElectron_MVA',
        'B_Run2_OSMuon_Dec',
        'B_Run2_OSMuon_Omega',
        'B_Run2_OSMuon_MVA',
    ]

def get_tex_decay(decay):
    if "Bu2JpsiK" == decay:
        tex_decay = r"$B^+ \to J/\psi K^+$"
    if "Bd2JpsiKst" == decay:
        tex_decay = r"$B^{*0} \to J/\psi K^*$"
    return tex_decay

def massfit(obs, masses, tex_decay, simulation, sim_fit, filename, df, compute_weights, generate_figures, obs_name, prefix=''):
    if not simulation: # Fix signal shape from MC
        with open(sim_fit) as f:
            _pars = json.load(f)
        alphaL = zfit.Parameter("alphaL", _pars["alphaL"]["value"],-5, 5.0, floating=False)
        nL = zfit.Parameter("nL", _pars["nL"]["value"], 0.1, 200, floating=False)
        alphaR = zfit.Parameter("alphaR", _pars["alphaR"]["value"], -5, 5.0, floating=False)
        nR = zfit.Parameter("nR", _pars["nR"]["value"], 0.1, 200, floating=False)
    else:
        alphaL = zfit.Parameter("alphaL", 1.4,-5, 5.0, floating=True)
        nL = zfit.Parameter("nL", 2.2, 0.1, 200, floating=True)
        alphaR = zfit.Parameter("alphaR", 1.4, -5, 5.0, floating=True)
        nR = zfit.Parameter("nR", 2.2, 0.1, 200, floating=True)

    # Initialise parameters
    # Signal DoubleCB
    mean = zfit.Parameter("mean", 5300, 5200, 5400)
    # g_mu = zfit.Parameter("g_mean", 5300, 5200, 5400)
    sigma = zfit.Parameter("sigma", 8, 0.1, 20)
    g_sigma = zfit.Parameter("g_sigma", 8, 0.1, 20)
    yield_signal = zfit.Parameter("yield_signal", len(df), 0, len(df))
    sig_frac1 = zfit.Parameter("sig_frac1", 0.5, 0, 1)
    # yield_gauss = zfit.Parameter("yield_signal_gauss", len(df), 0, len(df))

    double_cb = DoubleCB(mean, sigma, alphaL, nL, alphaR, nR, obs=obs)
    # signal_dcb = double_cb.create_extended(yield_signal)
    gauss =zfit.pdf.Gauss(obs=obs, mu=mean, sigma=g_sigma)
    # signal_gauss = gauss.create_extended(yield_gauss)
    model_sig = zfit.pdf.SumPDF(
        [double_cb, gauss], [sig_frac1])
    model_sig_ext = model_sig.create_extended(yield_signal)

    if not simulation:
        # Background (Exponential)
        lambda_ = zfit.Parameter("lambda", -0.001, -1.0, 0.0)
        yield_min = len(df) * 0.01 if compute_weights else len(df) * 0.5 #compute_weights also means BDT selection happened
        yield_bkg = zfit.Parameter("yield_bkg", yield_min, 0, len(df))

        # exp = Exponential(obs=obs, lambda_=lambda_)
        # comb=exp.create_extended(yield_bkg)

        c0 = zfit.Parameter("c0", 0.1)
        c1 = zfit.Parameter("c1", -0.2)
        c2 = zfit.Parameter("c2", 0.03)
        poly = zfit.pdf.Chebyshev(obs=obs, coeffs=[c0, c1, c2])
        comb_ext = poly.create_extended(yield_bkg)

        model = zfit.pdf.SumPDF([model_sig_ext, comb_ext])

    else:
        model = model_sig_ext


    # Convert data
    data = zfit.Data.from_numpy(obs=obs, array=masses)
    # Perform fit
    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    result.hesse()
    cov_matrix = result.covariance()
    params = result.params
    print(result)

    x_plot = np.linspace(mass_range[0], mass_range[1], 1000)
    signal_pdf_eval = model_sig.pdf(x_plot, norm_range=obs)
    sig_yield_val = params[yield_signal]['value']



    bins=100
    binwidth = (mass_range[1] - mass_range[0] )/bins
    signal_scaled = sig_yield_val * signal_pdf_eval * binwidth
    # gauss_scaled = gauss_yield_val * gauss_pdf_eval * binwidth
    total_signal_eval = signal_scaled

    if not simulation:
        background_pdf_eval = poly.pdf(x_plot, norm_range=obs)
        # background_pdf_eval = exp.pdf(x_plot, norm_range=obs)
        bkg_yield_val = params[yield_bkg]['value']
        bkg_scaled = bkg_yield_val * background_pdf_eval * binwidth
        total_pdf_eval = signal_scaled + bkg_scaled


    if generate_figures:
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

        if not simulation:
            ax1.plot(x_plot, bkg_scaled, label="Combinatorial", color="green", linestyle = "--", linewidth=2)
            ax1.plot(x_plot, total_pdf_eval, label="Total Fit", color='red',linewidth=3)



        # Scatter plot of data points with errors
        counts, bin_edges = np.histogram(masses, bins=bins, range=mass_range)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  
        errors = np.sqrt(counts)
        ax1.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', label="Data", markersize=1, elinewidth=1.0)

        # Compute fit curve

        ax1.plot(x_plot, total_signal_eval, label=tex_decay, color="blue", linestyle = "--", linewidth=2)




        ylabel = f"Events$~/~${binwidth}" + r"$[~\mathrm{MeV}/c^2]$"
        ax1.set_ylabel(ylabel)
        # ax1.set_yscale('log')
        ax1.legend()
        # Save the results
        params = result.params  # Get the fitted parameters

        # Extract parameter names and values
        fit_results = {param.name: {"value": info["value"], "error": info["hesse"]["error"]} for param, info in params.items()}

        # Save to JSON file
        outname = "mc_res.json" if cfg.simulation else "data_res.json"
        os.makedirs(outputdir, exist_ok=True)

        with open(join(outputdir, f'{outname}'), "w") as f:
            json.dump(fit_results, f, indent=4)

        # Compute residuals
        if cfg.simulation:
            total_pdf_eval = signal_scaled
        else:
            total_pdf_eval = signal_scaled + bkg_scaled
        total_fit_at_bin_centers = np.interp(bin_centers, x_plot, total_pdf_eval)
        residuals = (counts - total_fit_at_bin_centers) / errors
        ax2.axhline(0, color='black', linestyle='dashed')
        ax2.axhline(2, color='red', linestyle='dotted')
        ax2.axhline(-2, color='red', linestyle='dotted')
        ax2.scatter(bin_centers, residuals, color='black', marker='+')
        ax2.set_ylabel("Pull")

        if "Bu2JpsiK" in cfg.decayType:
            xlabel = r"$ m(B^+)~[\mathrm{MeV}/c^2]$"
        if "Bd2JpsiKst" in cfg.decayType:
            xlabel = r"$ m(B^{*0})~[\mathrm{MeV}/c^2]$"

        ax2.set_xlabel(xlabel)
        ax1.set_xlim(mass_range[0], mass_range[1])
        ax1.set_ylim(1, 1.1*np.max(counts))

        plt.tight_layout()
        plt.savefig(join(outputdir, filename))
        plt.close()

    if compute_weights:
        weights = compute_sweights(model, masses)

        df[f"{prefix}signal_weights"] = weights[yield_signal] 
        df[f"{prefix}background_weights"] = weights[yield_bkg] 
        del weights
        print(df)

        if generate_figures:
            plt.plot(masses, df[f"{prefix}signal_weights"], marker=".", linestyle="None", color="red", markersize=0.1, label="signal")
            plt.plot(masses, df[f"{prefix}background_weights"], marker=".", linestyle="None", color="green", markersize=0.1, label="background weights")
            plt.plot(masses, df[f"{prefix}background_weights"] + df[f"{prefix}signal_weights"], marker=".", linestyle="None", color="black", markersize=0.1, label="Sum of three")
            plt.xlabel("m($B^{+})~[MeV]/c^{2}$")
            plt.ylabel("weights")
            plt.legend()
            plt.savefig(join(outputdir,f"validate_sweights.png"))
            plt.close()

        #Calculate and save the pdf_ratio
        signal = model_sig_ext.pdf( df[obs_name], obs) * sig_yield_val
        bkg = comb_ext.pdf( df[obs_name], obs) * bkg_yield_val
        df["pdf_ratio"] = signal/bkg


            
            



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

    BDT = None
    if not cfg.simulation:
        #vars = run2_taggers_variables + ['RUNNUMBER', 'EVENTNUMBER',  "B_ID", 'entry', "FillNumber", "B_DTF_PV_Jpsi_MASS"] #f'{tagger}_TagDec', f'{tagger}_Eta',
        vars = ['RUNNUMBER', 'EVENTNUMBER', massname, 'B_ID']
        
        input_files = cfg.data_files
        if isinstance(input_files, str):
            input_files = [input_files]
        # Loop over all files
        dataframes = []

        bdt_features = vars_by_decay[cfg.decayType]

        with open(cfg.BDT, 'rb') as f:
            loaded_data = pickle.load(f)

        BDT = loaded_data['model']        
        cut   = loaded_data['cut']



        for i, f in enumerate(input_files):
            print(f"Reading input file: {f}")
            print(f'Megabites used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}', flush=True)

            filenumber = os.path.basename(f)[:-5]
            filenumber = int(filenumber[10:-9])

            with uproot.open(f) as _f:
                _df = _f[cfg.treename].arrays(vars+bdt_features, library="pd")
            _df.dropna(inplace=True)
            _df["SAMPLENUMBER"] = filenumber
            _df["event_entry"] = _df["SAMPLENUMBER"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            # _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'SAMPLENUMBER'], inplace=True)

            _df['signalness'] = BDT.predict_proba(_df[bdt_features].to_numpy())[:,1]
            _df.drop(columns=bdt_features, inplace=True)
            _df = _df.groupby("event_entry").first()

            dataframes.append(_df)
        #print(f'{pd.concat(singleTagger_dataframes).shape[0]}')
        df_data=pd.concat(dataframes, ignore_index=True)
        del dataframes

        print(df_data.shape)
        print(df_data.columns)        
    else:
        df_data = pd.DataFrame()
        input_files = cfg.sim_files
        for i, file in enumerate(input_files):
            print(f"Reading input file: {file}", flush=True)
            with uproot.open(file) as f:
                _df = f[cfg.treename].arrays([massname, 'RUNNUMBER', 'EVENTNUMBER'], library="pd")
            _df.dropna(inplace=True)
            _df["SAMPLENUMBER"] = i
            _df["event_entry"] = _df["SAMPLENUMBER"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'SAMPLENUMBER'], inplace=True)
            
            _df = _df.groupby("event_entry").first()

            df_data = pd.concat([df_data, _df], ignore_index = True)


    df_data = df_data.query(f'{cfg.obs} < {mass_range[1]} and {cfg.obs} > {mass_range[0]}')

    masses = df_data[massname].values
    obs = zfit.Space("mass", limits=mass_range)

    if isinstance(input_files, str):
        tex_decay = get_tex_decay(cfg.decayType)
    else:
        tex_decay = get_tex_decay(cfg.decayType)



    massfit(obs, masses, tex_decay, cfg.simulation, cfg.sim_fit, f"fit_res_before_cut.png", df_data, False, True, cfg.obs_name)

    if BDT is not None:
        pd.set_option('display.max_columns', 15)



        df_data = df_data[df_data['signalness'] > cut]
        df_data['BID_signal_weights'] = 0
        df_data['BID_background_weights'] = 0
        for id in df_data['B_ID'].unique():
            print(f'Calculating Sweights for BID={id}')
            df_fit = df_data[df_data['B_ID'] == id]
            masses  = df_fit[massname].values
            obs = zfit.Space("mass", limits=mass_range)
            massfit(obs, masses, tex_decay, cfg.simulation, cfg.sim_fit, f"fit_after_cut.png", df_fit, 
                    compute_weights= True, generate_figures= False, obs_name = cfg.obs_name, prefix='BID_')
            
            # print(df_data.head(10))

            # print(df_data.loc[df_data['B_ID'] == id])
            # print(df_fit['BID_signal_weights'].values)
            df_data.loc[df_data['B_ID'] == id, 'BID_signal_weights'] = df_fit['BID_signal_weights'].values
            df_data.loc[df_data['B_ID'] == id, 'BID_background_weights'] = df_fit['BID_background_weights'].values

        print(f'Calculating total Sweights')
        masses  = df_data[massname].values
        obs = zfit.Space("mass", limits=mass_range)
        massfit(obs, masses, tex_decay, cfg.simulation, cfg.sim_fit, f"fit_after_cut.png", df_data, True, True, cfg.obs_name)

        df_data.reset_index(inplace=True)
        # df_data.drop(columns=['event_entry'], inplace = True)

        #Save weighted dataframe to disk
        tree_dict = {col: np.array(df_data[col]) for col in df_data.columns}
        print(df_data.columns)
        del df_data
        print('Writing file to disk')


        with uproot.recreate(join(outputdir, 'weights.root')) as f:
            f['DecayTree'] = tree_dict


    print('Mass fit script ended')
    