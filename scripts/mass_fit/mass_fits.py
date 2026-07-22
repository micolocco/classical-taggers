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
from hepstats.splot import compute_sweights

import datetime
import psutil
from scripts.train_BDT import KFoldBDT
import pickle
import yaml
from pprint import pprint
import traceback
from matplotlib.lines import Line2D

from scripts.mass_fit.model_class import BsMassModel, GenericMassModel


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


def plot_mass_fit(model_builder, masses, bins, mass_range, model, params, yield_signal, yield_bkg, outputdir, filename, xlabel, tex_decay, is_simulation):
    print(f'Generating figures for {tex_decay} fit', flush=True)

    x_plot, binwidth, counts, bin_centers, errors = model_builder._common_plot_arrays(masses, bins, mass_range)
    ylabel = f"Events$~/~${binwidth}" + r"$[~\mathrm{MeV}/c^2]$"

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1]}, sharex=True)
    total_pdf_eval = model_builder.plot_mass_fit(ax1, x_plot, binwidth, model, params, yield_signal, yield_bkg)

    ax1.errorbar(bin_centers, counts, yerr=errors, fmt="o", color="black", label="MC Data" if is_simulation else "Data", markersize=1, elinewidth=1.0)
    ax1.set_ylabel(ylabel)
    ax1.set_yscale("log")
    ax1.legend()

    total_fit_at_bin_centers = np.interp(bin_centers, x_plot, total_pdf_eval)
    residuals = (counts - total_fit_at_bin_centers) / errors
    ax2.axhline(0, color="black", linestyle="dashed")
    ax2.axhline(2, color="red", linestyle="dotted")
    ax2.axhline(-2, color="red", linestyle="dotted")
    ax2.bar(bin_centers, residuals, width=binwidth, color="gray", alpha=0.5, label="Residuals")
    ax2.set_ylim(-5, 5)
    ax2.set_ylabel("Pull")
    ax2.set_xlabel(xlabel)
    ax1.set_xlim(mass_range[0], mass_range[1])

    y_min = np.min(counts)
    if y_min <= 2:
        y_min = 2
    ax1.set_ylim(y_min * 0.66, np.max(counts) * 2)

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
    if not simulation:  # Fix signal tail shape from MC
        with open(sim_fit) as f:
            _pars = json.load(f)
        mc_tail_parameters = {
            "alphaL": zfit.Parameter("alphaL", _pars["alphaL"]["value"], -5, 5.0, floating=False),
            "nL": zfit.Parameter("nL", _pars["nL"]["value"], 0.01, 200.0, floating=False),
            "alphaR": zfit.Parameter("alphaR", _pars["alphaR"]["value"], -5, 5.0, floating=False),
            "nR": zfit.Parameter("nR", _pars["nR"]["value"], 0.01, 200.0, floating=False),
        }
    else:
        mc_tail_parameters = None

    model_builder = GenericMassModel.from_decay_type(obs, tex_decay, simulation, is_selected, len(df), mc_tail_parameters)
    fit_context = model_builder.build()
    model = fit_context["model"]
    yield_signal = fit_context["yield_signal"]
    yield_bkg = fit_context["yield_bkg"]


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


    if generate_figures:
        plot_mass_fit(model_builder, masses, bins, mass_range, model, params, yield_signal, yield_bkg, outputdir, prefix + filename, xlabel, tex_decay, simulation)
                
    if compute_weights:

        if isinstance(model_builder, BsMassModel) and not simulation:
            # In case of Bs, refit to only the region relevant for Bs
            lower_bound = 5300

            bins = int(bins * (mass_range[1] - lower_bound) / (mass_range[1] - mass_range[0]))
            df = df.loc[df[obs_name] > lower_bound]
            masses = df[obs_name].values

            bs_context = model_builder.build_bs_region_context(fit_context["signal_params"], yield_signal, yield_bkg, mass_range, lower_bound=lower_bound)
            model = bs_context["model"]
            obs_restricted = bs_context["obs"]

            data = zfit.Data.from_numpy(obs=obs_restricted, array=masses)
            # Perform fit
            nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
            minimizer = zfit.minimize.Minuit(gradient=False, tol=1e-4, verbosity=5)
            result = minimizer.minimize(nll)
            result.hesse()
            params = result.params  # Get the fitted parameters
            print(result)

            plot_mass_fit(model_builder, masses, bins, (lower_bound, mass_range[1]), model, params, yield_signal, yield_bkg, outputdir, 'Bs_region_' + prefix + filename, xlabel, tex_decay, simulation)


        print('computing SWeights', flush=True)
        print(masses.shape)
        weights = compute_sweights(model, masses)
        print(weights)

        df[f"{prefix}signal_weights"]     = weights[yield_signal] 
        df[f"{prefix}background_weights"] = weights[yield_bkg] 

        #Plot the weighted mass spectra
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
    