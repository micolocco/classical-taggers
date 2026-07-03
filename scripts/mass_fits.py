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
import traceback
from matplotlib.lines import Line2D


def get_tex_decay(decay):
    if "Bu2JpsiK" == decay:
        tex_decay = r"$B^+ \to J/\psi K^+$"
        xlabel = r"$ m(J/\psi K^{\pm})~[\mathrm{MeV}/c^2]$"
    elif "Bd2JpsiKst" == decay:
        tex_decay = r"$B^{0} \to J/\psi K^*$"
        xlabel = r"$ m(J/\psi K^{*})~[\mathrm{MeV}/c^2]$"
    elif "Bs2JpsiKst" == decay:
        tex_decay = r"$B^{0}_{s} \to J/\psi K^*$"
        xlabel = r"$ m(J/\psi K^{*})~[\mathrm{MeV}/c^2]$"
    else:
        raise ValueError(f"Unknown decay type: {decay}")
    return tex_decay, xlabel

def fit_valid(result):
    if not result.valid:
        return [False, "fit is not valid, zfit reports invalid fit"]
    if result.edm > 1e-3:
        return [False, "fit is not valid, High edm"]
    
    relative_unc = [result.params[param]['hesse']['error'] / abs(result.params[param]['value']) if result.params[param]['value'] != 0 else np.nan for param in result.params]
    if any(unc > 0.5 for unc in relative_unc):
        return [False, "fit is not valid, High relative uncertainty"]
    return [True, "Fit is valid"]


def plot_mass_fit(masses, obs, bins, mass_range, is_simulation, model, use_secondary, 
                  params, yield_signal, yield_bkg, tex_decay, xlabel, outputdir, filename):
    print(f'Generating figures for {tex_decay} fit', flush=True)
    x_plot = np.linspace(mass_range[0], mass_range[1], 1000)
    binwidth = (mass_range[1] - mass_range[0] )/bins
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    if is_simulation:
        signal = model
    else:
        signal = model.models[0]
        background = model.models[1]

    signal_pdf_eval = signal.pdf(x_plot, norm_range=obs)
    sig_yield_val = params[yield_signal]['value']
    signal_scaled = sig_yield_val * signal_pdf_eval * binwidth
    total_pdf_eval = signal_scaled

    if not is_simulation:
        if not use_secondary:
            background_pdf_eval = background.pdf(x_plot, norm_range=obs)
            bkg_yield_val = params[yield_bkg]['value']
        else:
            secondary_model = background.models[0]
            comb_model = background.models[1]
            background_pdf_eval = comb_model.pdf(x_plot, norm_range=obs)
            bkg_yield_val = params[yield_bkg]['value'] * (1 - params["frac_secondary"]["value"])

            secondary_pdf_eval = secondary_model.pdf(x_plot, norm_range=obs)
            secondary_scaled = params[yield_bkg]['value'] * params["frac_secondary"]["value"] * secondary_pdf_eval * binwidth 

            total_pdf_eval += secondary_scaled    

            secondary_label = tex_decay.replace("B^{0} ", "B_{s}")

        bkg_scaled = bkg_yield_val * background_pdf_eval * binwidth

        total_pdf_eval += bkg_scaled



    # Plot the results
    if is_simulation:
        ax1.plot(x_plot, signal_scaled,  label=tex_decay, color="blue", linestyle = "--", linewidth=2)


    if not is_simulation:
        if use_secondary:
            if r"$B^{0} \to" in tex_decay:
                ax1.fill_between(x_plot,bkg_scaled, bkg_scaled+secondary_scaled, label=secondary_label, color="goldenrod", linewidth=2)
                ax1.plot(x_plot, signal_scaled,  label=tex_decay, color="blue", linestyle = "--", linewidth=2)
            else:
                ax1.fill_between(x_plot,bkg_scaled, bkg_scaled+signal_scaled, label=secondary_label, color="goldenrod", linewidth=2)
                ax1.plot(x_plot, secondary_scaled, label=tex_decay.replace("B^{0}_{s}", "B^{0}"), color="blue", linestyle = "--", linewidth=2)

        ax1.plot(x_plot, total_pdf_eval, label="Total Fit", color='darkred',linewidth=3)
        ax1.fill_between(x_plot, bkg_scaled, label="Combinatorial", color="lightgray", linewidth=2)

    # Scatter plot of data points with errors
    counts, bin_edges = np.histogram(masses, bins=bins, range=mass_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  
    errors = np.sqrt(counts)
    lab = "Data" if not is_simulation else "MC Data"
    ax1.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', label=lab, markersize=1, elinewidth=1.0)

    # Compute fit curve

    ylabel = f"Events$~/~${binwidth}" + r"$[~\mathrm{MeV}/c^2]$"
    ax1.set_ylabel(ylabel)
    ax1.set_yscale('log')
    ax1.legend()

    total_fit_at_bin_centers = np.interp(bin_centers, x_plot, total_pdf_eval)
    residuals = (counts - total_fit_at_bin_centers) / errors
    ax2.axhline(0, color='black', linestyle='dashed')
    ax2.axhline(2, color='red', linestyle='dotted')
    ax2.axhline(-2, color='red', linestyle='dotted')
    ax2.bar(bin_centers, residuals, width=binwidth, color='gray', alpha=0.5, label="Residuals")
    ax2.set_ylim(-5, 5)        
    ax2.set_ylabel("Pull")

    ax2.set_xlabel(xlabel)
    ax1.set_xlim(mass_range[0], mass_range[1])

    y_min = np.min(counts) #if not simulation else np.min(signal_scaled)
    if y_min <= 2:
        y_min = 2
    ax1.set_ylim(y_min*0.66, np.max(counts)*2)

    plt.tight_layout()
    plt.savefig(join(outputdir, filename))
    print(f'Figure saved to {join(outputdir, filename)}', flush=True)
    plt.close()

    # Plot individual PDF contributions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_plot, signal_scaled, label=tex_decay, color="blue", linewidth=2.5)
    if not is_simulation:
        ax.plot(x_plot, bkg_scaled, label="Combinatorial", color="gray", linewidth=2.5)
        if use_secondary:
            ax.plot(x_plot, secondary_scaled, label=secondary_label, color="goldenrod", linewidth=2.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(outputdir, filename.replace(".pdf", "_pdf_contributions.png")))
    print(f'PDF contributions plot saved to {join(outputdir, filename.replace(".pdf", "_pdf_contributions.png"))}', flush=True)
    plt.close()

    #Plot the Differences in counts and the Fit function
    diffs = counts - total_fit_at_bin_centers
    plt.plot(bin_centers, diffs)
    plt.xlabel(xlabel)
    plt.ylabel("Absolute Fit Error")
    plt.tight_layout()
    plt.savefig(join(outputdir, filename.replace(".pdf", "_Abs_Fit_error.png")))
    plt.close()

    diffs = (counts - total_fit_at_bin_centers)/counts
    plt.plot(bin_centers, diffs)
    plt.xlabel(xlabel)
    plt.ylabel("Relative Fit Error")
    plt.tight_layout()
    plt.savefig(join(outputdir, filename.replace(".pdf", "_Rel_Fit_error.png")))
    plt.close()




def massfit(obs, masses, tex_decay, xlabel, outname, simulation, sim_fit, filename, df, 
            compute_weights, generate_figures, obs_name, prefix='', is_selected = True, bins = 100):
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
    # Signal DoubleCB B0 or B^pm
    mean     = zfit.Parameter("mean_bd_or_bu",   5275.0, 5250.0, 5350)
    sigmaL   = zfit.Parameter("sigmaL",   12.0,    0.1,   30)
    sigmaR   = zfit.Parameter("sigmaR",   12.0,    0.1,   30)
    g_sigma  = zfit.Parameter("g_sigma",   6.0,    0.1,   30)
    sig_frac = zfit.Parameter("sig_frac",  0.5,    0.0,    1)
    
    double_cb = GeneralizedCB(obs=obs, mu=mean, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, nl=nL, alphar=alphaR, nr=nR)
    gauss =zfit.pdf.Gauss(obs=obs, mu=mean, sigma=g_sigma)
    model_Bpm_or_B0 = zfit.pdf.SumPDF([double_cb, gauss], [sig_frac])

    #DoubleCB for Bs if applicable
    mean_bs = zfit.ComposedParameter("mean_bs", lambda mean: mean + 87.45, params=mean)

    double_cb_bs = GeneralizedCB(obs=obs, mu=mean_bs, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, nl=nL, alphar=alphaR, nr=nR)
    gauss_bs = zfit.pdf.Gauss(obs=obs, mu=mean_bs, sigma=g_sigma)
    model_Bs = zfit.pdf.SumPDF([double_cb_bs, gauss_bs], [sig_frac])
    



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
            comb_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1, c2, c3, c4])
        else:
            lambd = zfit.Parameter("lambda", -0.01, -1, -5e-4) 


            comb_model = zfit.pdf.Exponential(obs=obs, lambda_=lambd)
    else:
        yield_bkg = None
            


    # Various combinations of models for the different cases of decay, simulation/Data, and selected/non-selected
    # Horrible if-else structure, having this many different fits in one file is not ideal
    yield_signal = zfit.Parameter("yield_signal", len(df), 0, len(df)*1.01)
    if simulation:
        if r"$B^{0}_{s}" in tex_decay:
            signal_ext = model_Bs.create_extended(yield_signal)
            model = signal_ext
        else:
            signal_ext = model_Bpm_or_B0.create_extended(yield_signal)
            model = signal_ext
    else:
        if r"$B^+" in tex_decay or (r"$B^{0}" in tex_decay and not is_selected):
            background_ext = comb_model.create_extended(yield_bkg)

            signal_ext = model_Bpm_or_B0.create_extended(yield_signal)
            model = zfit.pdf.SumPDF([signal_ext, background_ext])
        elif r"$B^{0}" in tex_decay or r"$B^{0}_{s}" in tex_decay:
            start_frac = 0.1 if r"$B^{0} \to" in tex_decay else 0.4
            frac_secondary = zfit.Parameter("frac_secondary", start_frac, 0, 1)
            # yield_bs = zfit.Parameter("yield_Bs", len(df), 0, len(df)*1.01)


            if r"$B^{0} \to" in tex_decay:
                signal_ext = model_Bpm_or_B0.create_extended(yield_signal)
                secondary_model = model_Bs


                background_model = zfit.pdf.SumPDF([secondary_model, comb_model], [frac_secondary])
                
                background_ext = background_model.create_extended(yield_bkg)

            else:
                signal_ext = model_Bs.create_extended(yield_signal)

                secondary_model = model_Bpm_or_B0

                background_model = zfit.pdf.SumPDF([secondary_model, comb_model], [frac_secondary])
                background_ext = background_model.create_extended(yield_bkg)


            model = zfit.pdf.SumPDF([signal_ext, background_ext],)


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
    
    cor_matrix = result.correlation()
    print(f"correlation matrix:\n{cor_matrix}", flush=True)
    print(f'Result message: {result.message}', flush=True)
    # Save the results
    params = result.params  # Get the fitted parameters

    # Extract parameter names and values
    try:
        fit_results = {param.name: {"value": info["value"], "error": info["hesse"]["error"]} for param, info in params.items()}

        # Save to JSON file
        os.makedirs(outputdir, exist_ok=True)

        with open(join(outputdir, f'{outname}.json'), "w") as f:
            json.dump(fit_results, f, indent=4)
    except Exception as e:
        # Print entire traceback for debugging
        print("Error while saving fit results to JSON:")
        traceback.print_exc()


    use_secondary = not simulation and r"$B^{0}" in tex_decay and is_selected
    if generate_figures:
        plot_mass_fit(masses, obs, bins, mass_range, simulation, model, use_secondary, 
                      params, yield_signal, yield_bkg, tex_decay, xlabel, outputdir, prefix + filename)
                
    if compute_weights:

        if r"$B^{0}_{s} \to" in tex_decay and not simulation:
            #In case of Bs, refit to only the region relevant for Bs 
            lower_bound = 5300

            bins = int(bins*(mass_range[1] - lower_bound) / (mass_range[1] - mass_range[0]))
            df = df.loc[df[obs_name] > lower_bound]
            masses = df[obs_name].values

            # binwidth = (mass_range[1] - lower_bound ) / bins


            mean.floating     = False
            sigmaL.floating   = False
            sigmaR.floating   = False
            g_sigma.floating = False
            sig_frac.floating = False
            lambd.floating    = False


            obs_restricted = zfit.Space("mass", limits=(lower_bound, mass_range[1]))



            double_cb = GeneralizedCB(obs=obs_restricted, mu=mean, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, nl=nL, alphar=alphaR, nr=nR)
            gauss =zfit.pdf.Gauss(obs=obs_restricted, mu=mean, sigma=g_sigma)
            double_cb_bs = GeneralizedCB(obs=obs_restricted, mu=mean_bs, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, nl=nL, alphar=alphaR, nr=nR)
            gauss_bs = zfit.pdf.Gauss(obs=obs_restricted, mu=mean_bs, sigma=g_sigma)

            comb_model = zfit.pdf.Exponential(obs=obs_restricted, lambda_=lambd)
            model_Bs = zfit.pdf.SumPDF([double_cb_bs, gauss_bs], [sig_frac])
    
            model_Bpm_or_B0 = zfit.pdf.SumPDF([double_cb, gauss], [sig_frac])
            signal_ext = model_Bs.create_extended(yield_signal)

            secondary_model = model_Bpm_or_B0

            background_model = zfit.pdf.SumPDF([secondary_model, comb_model], [frac_secondary])
            background_ext = background_model.create_extended(yield_bkg)
            model = zfit.pdf.SumPDF([signal_ext, background_ext],)


            data = zfit.Data.from_numpy(obs=obs_restricted, array=masses)
            # Perform fit
            nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
            minimizer = zfit.minimize.Minuit(gradient=False, tol= 1e-4, verbosity=5)
            result = minimizer.minimize(nll)
            result.hesse()
            params = result.params  # Get the fitted parameters
            print(result)


            plot_mass_fit(masses, obs_restricted, bins, (lower_bound, mass_range[1]), simulation, model, use_secondary, 
                          params, yield_signal, yield_bkg, tex_decay, xlabel, outputdir, 'Bs_region_' + prefix + filename)


        print('computing SWeights', flush=True)
        print(masses.shape)
        weights = compute_sweights(model, masses)
        print(weights)

        df[f"{prefix}signal_weights"]     = weights[yield_signal] 
        df[f"{prefix}background_weights"] = weights[yield_bkg] 


        if generate_figures:
            plt.figure(figsize=(10, 6))
            plt.hist(
                masses,
                bins=100,
                weights=df[f"{prefix}signal_weights"],
                histtype="step",
                linewidth=2,
                color="red",
                label="signal-weighted",
                alpha=0.7
            )
            plt.hist(
                masses,
                bins=100,
                weights=df[f"{prefix}background_weights"],
                histtype="step",
                linewidth=2,
                color="green",
                label="background-weighted",
                alpha=0.7
            )
            plt.xlabel(xlabel)
            plt.ylabel("Weighted candidates")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(join(outputdir, prefix + "weighted_mass_spectra.png"))
            plt.close()

        
        del weights
        if generate_figures:
            plt.plot(masses, df[f"{prefix}signal_weights"], marker=".", linestyle="None", color="red", markersize=1, label="signal")
            plt.plot(masses, df[f"{prefix}background_weights"], marker=".", linestyle="None", color="green", markersize=1, label="background weights")
            weight_sum = df[f"{prefix}signal_weights"] + df[f"{prefix}background_weights"]

            plt.plot(masses, weight_sum, marker=".", linestyle="None", color="black", markersize=1, label="Sum of three")
            plt.xlabel(xlabel)
            plt.ylabel("weights")
            # use line-only entries in the legend (points remain as plotted)
            handles = [Line2D([0], [0], color='red', linestyle='-', label='signal'),
                       Line2D([0], [0], color='green', linestyle='-', label='background weights')]
            handles.append(Line2D([0], [0], color='black', linestyle='-', label='Sum of three'))
            plt.legend(handles=handles)
            plt.grid(True, alpha=0.3)
            plt.savefig(join(outputdir, prefix + f"validate_sweights.png"))
            plt.close()

    print(f"Fit ended with status: {result.status}", flush=True)
    return df



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
    parser.add_argument('--obs_name', help='Name of observable', type=str, default="B_DTF_PV_Jpsi_MASS")
    parser.add_argument('--num_threads', help='Number of threads to use for the fits', type=int, default=1)
    parser.add_argument('--selected', action="store_true", help='Whether the input data has already been selected with the BDT cut.')

    cfg = parser.parse_args()
    pprint(cfg)

    massname = cfg.obs_name
    mass_range = (int(cfg.range[0]), int(cfg.range[1]))
    outputdir = cfg.output
    os.makedirs(outputdir, exist_ok=True)

    zfit.run.set_n_cpu(n_cpu=cfg.num_threads)


    print(f'Reading files started on {datetime.datetime.now().strftime("%H:%M:%S")}')
    vars = ['file_id', 'RUNNUMBER', 'EVENTNUMBER', massname, 'candidate_index']

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

        print(f'File read. Number of events: {len(_df["event_entry"].unique())}, number of candidates: {len(_df)}', flush=True)


        if df_data is None:
            df_data = _df
        else:
            df_data = pd.concat([df_data, _df], ignore_index=True)

    print(f'Reading files ended on {datetime.datetime.now().strftime("%H:%M:%S")}')


    df_data = df_data.query(f'{massname} < {mass_range[1]} and {massname} > {mass_range[0]}')

    masses = df_data[massname].values
    obs = zfit.Space("mass", limits=mass_range)


    tex_decay, xlabel = get_tex_decay(cfg.decay_type)
    selected_string = f'selected' if cfg.selected else 'non_selected'
    filename = f'event_{selected_string}_fit'
    prefix = '' if cfg.selected else 'non_selected_'
    df_data = massfit(obs, masses, tex_decay, xlabel, filename, cfg.simulation, cfg.sim_fit, f"fit_res_{selected_string}.pdf", df_data, 
            compute_weights= not cfg.simulation, generate_figures=True, obs_name=cfg.obs_name, is_selected = cfg.selected, prefix=prefix, )
    
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
    