import zfit
import uproot
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("LHCb2")
import argparse
from hepstats.splot import compute_sweights
import os
import json

import zfit
from zfit.models.physics import DoubleCB
from zfit.models.functor import SumPDF
from hepstats.splot import compute_sweights


def signalname_from_filename(file):
    if "Bu2JpsiK" in file:
        signalname = r"$B^+ \to J/\psi K^+$"
    if "Bd2JpsiKst" in file:
        signalname = r"$B^{*0} \to J/\psi K^*$"
    return signalname
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add the sample weights to the data set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--selected', help='File with preselection applied')
    parser.add_argument('--data_fit_model', help='The saved Model of the data massfits', type=str)
    parser.add_argument('--sim_fit_model', help='The saved Model of the simulation massfits', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--decayType', help='Event decay', type=str)
    parser.add_argument('--obs_name', help='Name of observable', type=str, default="B_DTF_PV_Jpsi_MASS")
    parser.add_argument('--range', help='Observable range', nargs="+")
    parser.add_argument('--out_path', help='Where fit results and plots will be stored', type=str)


    cfg = parser.parse_args()

    mass_range = (int(cfg.range[0]), int(cfg.range[1]))
    id = os.path.basename(cfg.selected)[:-5]

    #Load Dataset
    with uproot.open(cfg.selected) as _f:
        df_data = _f[cfg.treename].arrays(library="pd")
    df_data.dropna(inplace=True)
    df_data["event_entry"] =  df_data["RUNNUMBER"].astype(str) + "_" + df_data["EVENTNUMBER"].astype(str)

    df_data = df_data.query(f'{cfg.obs_name} < {mass_range[1]} and {cfg.obs_name} > {mass_range[0]}')


    df_fit = df_data.groupby("event_entry").first()
    df_fit.index.name = 'event_entry'

    cols_to_keep = ['event_entry', cfg.obs_name]

    df_fit.drop(df_fit.columns.difference(cols_to_keep), axis=1, inplace=True)

    masses = df_fit[cfg.obs_name].values

    obs = zfit.Space("mass", limits=mass_range)

    #Load results from massfit
    with open(cfg.data_fit_model) as f:
        _pars = json.load(f)
    mean    = zfit.Parameter("mean",    _pars["mean"]["value"],    floating=False)
    sigma   = zfit.Parameter("sigma",   _pars["sigma"]["value"],   floating=False)
    g_sigma = zfit.Parameter("g_sigma", _pars["g_sigma"]["value"], floating=False)
    c0      = zfit.Parameter("c0",      _pars["c0"]["value"],      floating=False)
    c1      = zfit.Parameter("c1",      _pars["c1"]["value"],      floating=False)
    c2      = zfit.Parameter("c2",      _pars["c2"]["value"],      floating=False)


    with open(cfg.sim_fit_model) as f:
        _pars = json.load(f)
    alphaL = zfit.Parameter("alphaL", _pars["alphaL"]["value"], floating=False)
    nL     = zfit.Parameter("nL",     _pars["nL"]["value"],     floating=False)
    alphaR = zfit.Parameter("alphaR", _pars["alphaR"]["value"], floating=False)
    nR     = zfit.Parameter("nR",     _pars["nR"]["value"],     floating=False)


    #define floating parameters
    yield_signal = zfit.Parameter("yield_signal", len(df_fit), 0, len(df_fit))
    sig_frac1 = zfit.Parameter("sig_frac1", 0.5, 0, 1)
    yield_bkg = zfit.Parameter("yield_bkg", len(df_fit) * 0.2, 0, len(df_fit))


    #Build Model
    double_cb = DoubleCB(mean, sigma, alphaL, nL, alphaR, nR, obs=obs)
    gauss =zfit.pdf.Gauss(obs=obs, mu=mean, sigma=g_sigma)
    model_sig = zfit.pdf.SumPDF([double_cb, gauss], [sig_frac1])
    model_sig_ext = model_sig.create_extended(yield_signal)
    poly = zfit.pdf.Chebyshev(obs=obs, coeffs=[c0, c1, c2])
    comb_ext = poly.create_extended(yield_bkg)

    model = zfit.pdf.SumPDF([model_sig_ext, comb_ext])

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

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    bins=100
    binwidth = (mass_range[1] - mass_range[0] )/bins
    # Scatter plot of data points with errors
    counts, bin_edges = np.histogram(masses, bins=bins, range=mass_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  
    errors = np.sqrt(counts)
    ax1.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', label="Data", markersize=1, elinewidth=1.0)

    # Compute fit curve
    x_plot = np.linspace(mass_range[0], mass_range[1], 1000)
    signal_pdf_eval = model_sig.pdf(x_plot, norm_range=obs)
    sig_yield_val = params[yield_signal]['value']
    # gauss_pdf_eval = gauss.pdf(x_plot,norm_range=obs)
    # gauss_yield_val = params[yield_gauss]['value']


    sig_yield_val = params[yield_signal]['value']
    signal_scaled = sig_yield_val * signal_pdf_eval * binwidth
    # gauss_scaled = gauss_yield_val * gauss_pdf_eval * binwidth
    total_signal_eval = signal_scaled

    ax1.plot(x_plot, total_signal_eval, label=signalname_from_filename(cfg.selected), color="blue", linestyle = "--", linewidth=2)


    background_pdf_eval = poly.pdf(x_plot, norm_range=obs)
    # background_pdf_eval = exp.pdf(x_plot, norm_range=obs)
    bkg_yield_val = params[yield_bkg]['value']
    bkg_scaled = bkg_yield_val * background_pdf_eval * binwidth
    total_pdf_eval = signal_scaled + bkg_scaled
    ax1.plot(x_plot, bkg_scaled, label="Combinatorial", color="green", linestyle = "--", linewidth=2)
    ax1.plot(x_plot, total_pdf_eval, label="Total Fit", color='red',linewidth=3)

    ylabel = f"Events$~/~${binwidth}" + r"$[~\mathrm{MeV}/c^2]$"
    ax1.set_ylabel(ylabel)
    ax1.legend()


    # Compute residuals

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
    ax1.set_ylim(0, 1.1*np.max(counts))

    plt.tight_layout()
    plot_path = join(cfg.out_path, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(join(plot_path, f'{id}_fit_res.png'))
    plt.close()





    #Calculate and save the sweights
    weights = compute_sweights(model, masses)

    df_fit["signal_weights"] = weights[yield_signal] 
    df_fit["background_weights"] = weights[yield_bkg] 

    plt.plot(masses, df_fit["signal_weights"], marker=".", linestyle="None", color="red", markersize=0.1, label="signal")
    plt.plot(masses, df_fit["background_weights"], marker=".", linestyle="None", color="green", markersize=0.1, label="background weights")
    plt.plot(masses, df_fit["background_weights"] + df_fit["signal_weights"], marker=".", linestyle="None", color="black", markersize=0.1, label="Sum of three")
    plt.xlabel("m($B^{+})~[MeV]/c^{2}$")
    plt.ylabel("weights")
    plt.legend()
    plt.savefig(join(plot_path,f"validate_sweights_{id}.png"))
    plt.close()

    #Calculate and save the pdf_ratio
    signal = model_sig_ext.pdf( df_fit[cfg.obs_name], obs) * sig_yield_val
    bkg = comb_ext.pdf( df_fit[cfg.obs_name], obs) * bkg_yield_val
    df_fit["pdf_ratio"] = signal/bkg

    df_data = df_data.merge(df_fit.reset_index(), on=cols_to_keep, how='left')
    df_data.drop(columns=['event_entry'], inplace = True)

    #Save weighted dataframe to disk
    tree_dict = {col: np.array(df_data[col]) for col in df_data.columns}


    with uproot.recreate(join(cfg.out_path, os.path.basename(cfg.selected))) as f:
        f['DecayTree'] = tree_dict