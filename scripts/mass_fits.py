import uproot
import numpy as np
import pandas as pd
import os
from os.path import join
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
import psutil
from scripts.train_BDT import KFoldBDT
import pickle
import yaml
from pprint import pprint


def get_tex_decay(decay):
    if "Bu2JpsiK" == decay:
        tex_decay = r"$B^+ \to J/\psi K^+$"
    if "Bd2JpsiKst" == decay:
        tex_decay = r"$B^{0} \to J/\psi K^*$"
    return tex_decay

def fit_valid(result):
    if not result.valid:
        return [False, "fit is not valid, zfit reports invalid fit"]
    if result.edm > 1e-3:
        return [False, "fit is not valid, High edm"]
    
    relative_unc = [result.params[param]['hesse']['error'] / abs(result.params[param]['value']) if result.params[param]['value'] != 0 else np.nan for param in result.params]
    if any(unc > 0.5 for unc in relative_unc):
        return [False, "fit is not valid, High relative uncertainty"]
    return [True, "Fit is valid"]


def massfit(obs, masses, tex_decay, outname, simulation, sim_fit, filename, df, compute_weights, generate_figures, obs_name, prefix='', is_selected = True):
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
        nL     = zfit.Parameter("nL",      1.5,  0.01,  15.0, floating=True)
        alphaR = zfit.Parameter("alphaR",  3.0,  0.0,    5.0, floating=True)
        nR     = zfit.Parameter("nR",      1.6,  0.01,  15.0, floating=True)

        






    # Initialise parameters
    # Signal DoubleCB
    mean     = zfit.Parameter("mean",   5300.0, 5200.0, 5400)
    sigmaL   = zfit.Parameter("sigmaL",   12.0,    0.1,   30)
    sigmaR   = zfit.Parameter("sigmaR",   12.0,    0.1,   30)
    g_sigma1 = zfit.Parameter("g_sigma1",  6.0,    0.1,   30)
    g_sigma2 = zfit.Parameter("g_sigma2", 10.0,    0.1,   30)
    sig_frac = zfit.Parameter("sig_frac",  0.5,    0.0,    1)
    g_frac1  = zfit.Parameter("g_frac1",   0.1,    0.0,    1)

    yield_signal = zfit.Parameter("yield_signal", len(df), 0, len(df)*1.01)
    
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
        yield_min = len(df) * 0.1 if is_selected else len(df) * 0.5
        yield_bkg = zfit.Parameter("yield_bkg", yield_min, 0, len(df))


        if not is_selected: # Pre BDT cuts the background is not exponentially shaped -> chebyshev
            c1 = zfit.Parameter("c1",  0.05)
            c2 = zfit.Parameter("c2", -0.2)
            c3 = zfit.Parameter("c3",  0.0)
            c4 = zfit.Parameter("c4",  0.01)
            bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1, c2, c3, c4])
        else:
            lambd = zfit.Parameter("lambda", -0.01, -1, -5e-4) 
            comb_model = zfit.pdf.Exponential(obs=obs, lambda_=lambd)

            if r"$B^{0}" in tex_decay: #For B0 decays: add a peaking background structure from Bs decays. same shape as signal just shifted and scaled
                mean_bs = zfit.ComposedParameter("mean_bs", lambda mean: mean + 87.45, params=mean)

                double_cb_bs = GeneralizedCB(obs=obs, mu=mean_bs, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, nl=nL, alphar=alphaR, nr=nR)
                gauss_bs1 = zfit.pdf.Gauss(obs=obs, mu=mean_bs, sigma=g_sigma1)
                gauss_bs2 = zfit.pdf.Gauss(obs=obs, mu=mean_bs, sigma=g_sigma2)

                gauss_bs = zfit.pdf.SumPDF([gauss_bs1, gauss_bs2], [g_frac1])
                

                model_bs = zfit.pdf.SumPDF([double_cb_bs, gauss_bs], [sig_frac])
                
                bs_bkg_frac = zfit.Parameter("yield_bs", 0.01, 0, 1)
                bkg_model = zfit.pdf.SumPDF([model_bs, comb_model], [bs_bkg_frac])
            else:
                bkg_model = comb_model



        comb_ext = bkg_model.create_extended(yield_bkg)

        model = zfit.pdf.SumPDF([model_sig_ext, comb_ext])

    else:
        model = model_sig_ext


    # Convert data
    data = zfit.Data.from_numpy(obs=obs, array=masses)
    # Perform fit
    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit(gradient=False, tol= 1e-4, verbosity=5)
    result = minimizer.minimize(nll)
    result.hesse()

    print(f"Initial fit converged: {result.converged} with edm: {result.edm}", flush=True)
    
    validity = fit_valid(result)
    print(f'Initial fit')
    print(validity[1], flush=True)
    print(result, flush=True)
    i = 0
    while not validity[0] and i < 3:  # If the fit is not valid or edm is too high, try again
        result = minimizer.minimize(nll, init=result)
        result.hesse()
        validity = fit_valid(result)
        print(f'Fit attempt {i+1}')
        print(validity[1], flush=True)
        print(result, flush=True)
        i += 1


    if not validity[0]:  # If the fit is not valid or edm is too high, try again
        result = minimizer.minimize(nll, init=result)
        result.hesse()
    
    cov_matrix = result.covariance()
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
        if r"$B^{0}" not in tex_decay or not is_selected:
            background_pdf_eval = bkg_model.pdf(x_plot, norm_range=obs)
            bkg_yield_val = params[yield_bkg]['value']
        else:
            background_pdf_eval = comb_model.pdf(x_plot, norm_range=obs)
            bkg_yield_val = params[yield_bkg]['value'] * (1 - params["yield_bs"]["value"])


        bkg_scaled = bkg_yield_val * background_pdf_eval * binwidth
        total_pdf_eval = signal_scaled + bkg_scaled
    


    if generate_figures:
        print(f'Generating figures for {tex_decay} fit', flush=True)
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

        if not simulation:
            if r"$B^{0}" in tex_decay and is_selected:
                bs_pdf_eval = model_bs.pdf(x_plot, norm_range=obs)
                bs_scaled = params[yield_bkg]['value'] * params["yield_bs"]["value"] * bs_pdf_eval * binwidth 

                total_pdf_eval += bs_scaled    
                

                ax1.fill_between(x_plot,bkg_scaled, bkg_scaled+bs_scaled, label=tex_decay.replace("B^{0}", "B_{s}"), color="goldenrod", linewidth=2)

            ax1.plot(x_plot, total_pdf_eval, label="Total Fit", color='darkred',linewidth=3)
            ax1.fill_between(x_plot, bkg_scaled, label="Combinatorial", color="lightgray", linewidth=2)



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
            if r"$B^{0}" in tex_decay and is_selected:
                total_pdf_eval += bs_scaled
        total_fit_at_bin_centers = np.interp(bin_centers, x_plot, total_pdf_eval)
        residuals = (counts - total_fit_at_bin_centers) / errors
        ax2.axhline(0, color='black', linestyle='dashed')
        ax2.axhline(2, color='red', linestyle='dotted')
        ax2.axhline(-2, color='red', linestyle='dotted')
        ax2.bar(bin_centers, residuals, width=binwidth, color='gray', alpha=0.5, label="Residuals")
        ax2.set_ylim(-5, 5)        
        ax2.set_ylabel("Pull")

        if r"$B^+ \to J/\psi K^+$" in tex_decay:
            xlabel = r"$ m(J/\psi K^{\pm})~[\mathrm{MeV}/c^2]$"
        elif r"$B^{0} \to J/\psi K^*$" in tex_decay:
            xlabel = r"$ m(J/\psi K^{*})~[\mathrm{MeV}/c^2]$"
        else:
            raise ValueError(f"Unknown decay type: {tex_decay}")

        ax2.set_xlabel(xlabel)
        ax1.set_xlim(mass_range[0], mass_range[1])

        y_min = np.min(bkg_scaled) if not simulation else np.min(signal_scaled)
        if y_min <= 2:
            y_min = 2
        ax1.set_ylim(y_min*0.66, np.max(counts)*2)

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
            plt.plot(masses, df[f"{prefix}signal_weights"], marker=".", linestyle="None", color="red", markersize=1, label="signal")
            plt.plot(masses, df[f"{prefix}background_weights"], marker=".", linestyle="None", color="green", markersize=1, label="background weights")
            plt.plot(masses, df[f"{prefix}background_weights"] + df[f"{prefix}signal_weights"], marker=".", linestyle="None", color="black", markersize=1, label="Sum of three")
            plt.xlabel("m($B^{+})~[MeV]/c^{2}$")
            plt.ylabel("weights")
            plt.legend()
            plt.savefig(join(outputdir, prefix + f"validate_sweights.pdf"))
            plt.close()

        #Calculate and save the pdf_ratio
        signal = model_sig_ext.pdf( df[obs_name], obs) * sig_yield_val
        bkg = comb_ext.pdf( df[obs_name], obs) * bkg_yield_val
        df["pdf_ratio"] = signal/bkg



    print(f"Fit ended with status: {result.status}", flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Massfits for B+ and B0 decays to extract sWeights',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input_files', nargs='+')
    parser.add_argument('--treename', help='TreeName of the input file', type=str, default="BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree")
    parser.add_argument('--range', help='Observable range', nargs="+")
    parser.add_argument('--output', help='Where fit results and plots will be stored', type=str)
    parser.add_argument('--simulation', action="store_true")
    parser.add_argument('--sim_fit', help="Fit results from mc")
    parser.add_argument('--decay_type', help='Decay used', type=str)
    # parser.add_argument('--BDT', help='Path of BDT for event selection') 
    parser.add_argument('--obs_name', help='Name of observable', type=str, default="B_DTF_PV_Jpsi_MASS")
    parser.add_argument('--num_threads', help='Number of threads to use for the fits', type=int, default=1)
    # parser.add_argument('--signal_class_features', help='Path to yaml file containing the features used for the signal classification BDT')
    parser.add_argument('--selected', action="store_true", help='Whether the input data has already been selected with the BDT cut.')

    cfg = parser.parse_args()
    pprint(cfg)

    massname = cfg.obs_name
    mass_range = (int(cfg.range[0]), int(cfg.range[1]))
    outputdir = cfg.output
    os.makedirs(outputdir, exist_ok=True)

    zfit.run.set_n_cpu(n_cpu=cfg.num_threads)


    print(f'Reading files started on {datetime.datetime.now().strftime("%H:%M:%S")}')
    vars = ['file_id', 'RUNNUMBER', 'EVENTNUMBER', massname, 'B_ID', 'candidate_index']

    if cfg.simulation:
        vars += ['B_BKGCAT']

    input_files = cfg.input_files
    df_data = None
    for i, f in enumerate(input_files):
        print(f"Reading input file {i+1}/{len(input_files)}: {f}")
        print(f'Megabites used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}', flush=True)
        with uproot.open(f) as _f:
            _df = _f[cfg.treename].arrays(vars, library="pd")
        _df.dropna(inplace=True)
        
        if cfg.simulation:
            _df = _df.query("B_BKGCAT == 0")  
        
        _df["event_entry"] = _df["file_id"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        _df['candidate_entry'] = _df['file_id'].astype(str) + "_" + _df['candidate_index'].astype(str)

        _df = _df.groupby("candidate_entry").first()
        _df.reset_index(inplace=True)

        if df_data is None:
            df_data = _df
        else:
            df_data = pd.concat([df_data, _df], ignore_index=True)

    print(f'Reading files ended on {datetime.datetime.now().strftime("%H:%M:%S")}')


    df_data = df_data.query(f'{massname} < {mass_range[1]} and {massname} > {mass_range[0]}')

    masses = df_data[massname].values
    obs = zfit.Space("mass", limits=mass_range)


    tex_decay = get_tex_decay(cfg.decay_type)
    selected_string = f'selected' if cfg.selected else 'non_selected'
    filename = f'event_{selected_string}_fit'
    prefix = '' if cfg.selected else 'non_selected_'
    massfit(obs, masses, tex_decay, filename, cfg.simulation, cfg.sim_fit, f"fit_res_{selected_string}.pdf", df_data, 
            compute_weights= not cfg.simulation, generate_figures=True, obs_name=cfg.obs_name, is_selected = cfg.selected, prefix=prefix)
    
    #Save weighted dataframe to disk
    tree_dict = {col: np.array(df_data[col]) for col in df_data.columns if col != 'event_entry' and col != 'candidate_entry'}
    print(tree_dict)

    for col in tree_dict:
        print(f'{col}: {tree_dict[col][0]} ({type(tree_dict[col][0])})')

    del df_data
    print('Writing file to disk')


    with uproot.recreate(join(outputdir, f'weights_{selected_string}.root')) as f:
        f['DecayTree'] = tree_dict


    print(f'Mass fit script ended on {datetime.datetime.now().strftime("%H:%M:%S")}')
    