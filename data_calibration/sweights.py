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
import zfit

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

def signalname_from_filename(file):
    if "Bu2JpsiK" in file:
        signalname = r"$B^+ \to J/\psi K^+$"
    if "Bd2JpsiKst" in file:
        signalname = r"$B^{*0} \to J/\psi K^*$"
    return signalname
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagged_prePath')
    parser.add_argument('--treename', help='TreeName of the input file', type=str, default="BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree")
    parser.add_argument('--obs', help='Observable to fit', type=str, default="B_DTF_PV_Jpsi_MASS")
    parser.add_argument('--range', help='Observable range', nargs="+")
    parser.add_argument('--output', help='Where fit results and plots will be stored', type=str)
    parser.add_argument('--simulation', action="store_true")
    parser.add_argument('--sim_fit', help="Fit results from mc")
    parser.add_argument('--sim_files', help="MC files", nargs="+")
    parser.add_argument('--tagger', help="tagger output to be saved", nargs="+")
    parser.add_argument('--decayType', help='Decay used for the calibration', type=str)
    parser.add_argument('--cut', help='Cut desired', type=str)
    parser.add_argument('--features', help='Input features used for NN training') 

    # to do parse background model 

    cfg = parser.parse_args()
    # pprint(cfg)

    massname = cfg.obs
    mass_range = (int(cfg.range[0]), int(cfg.range[1]))
    outputdir = join(cfg.output, "mc_fit") if cfg.simulation else join(cfg.output, "data_fit")
    os.makedirs(outputdir, exist_ok=True)

    if not cfg.simulation:
        taggers_dataframes = []  # List to store DataFrames for each tagger
        for tagger in cfg.tagger:
            print(tagger)
            vars = run2_taggers_variables + ['RUNNUMBER', 'EVENTNUMBER', f'{tagger}_TagDec', f'{tagger}_Eta', "B_ID", 'entry', "FillNumber", "B_DTF_PV_Jpsi_MASS", "B_DTF_PV_CTAU"]
            input_path = os.path.join(cfg.tagged_prePath, cfg.decayType, tagger, cfg.cut, cfg.features, '*.root')
            input_files = glob.glob(input_path)
            # Loop over all files
            singleTagger_dataframes = []
            for i, f in enumerate(input_files):
                # print(f"Reading input file: {f}")
                with uproot.open(f) as _f:
                    _df = _f[cfg.treename].arrays(vars, library="pd")
                _df.dropna(inplace=True)
                _df["SAMPLENUMBER"] = i
                _df["event_entry"] = _df["SAMPLENUMBER"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
                _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'SAMPLENUMBER'], inplace=True)

                singleTagger_dataframes.append(_df)
            #print(f'{pd.concat(singleTagger_dataframes).shape[0]}')
            df_merged=pd.concat(singleTagger_dataframes, ignore_index=True)
            df_merged = df_merged.groupby("event_entry").first()
            taggers_dataframes.append(df_merged)

        df_data = taggers_dataframes[0]
        print(df_data.shape)
        print(df_data.columns)
        print('Dataframe shape must have same row number for a correct combination! Check it!')
        for single_df in taggers_dataframes[1:]:
            print(single_df.shape)
            df_data = pd.merge(df_data, single_df, on=['event_entry', "B_ID", massname, "entry", "FillNumber", "B_DTF_PV_CTAU"]+run2_taggers_variables, how='outer')   
            print(f'total:{df_data.shape}')
        # Merge all DataFrames on the common columns
        # df_data = df_data.query("FillNumber < 10056 and FillNumber > 9982") # for block1 selection
        
    else:
        df_data = pd.DataFrame()
        input_files = cfg.sim_files
        for file in input_files:
            with uproot.open(file) as f:
                _df = f[cfg.treename].arrays([massname], library="pd")
            df_data = pd.concat([df_data, _df], ignore_index = True)


    df_data = df_data.query(f'{cfg.obs} < {mass_range[1]} and {cfg.obs} > {mass_range[0]}')

    masses = df_data[massname].values
    obs = zfit.Space("mass", limits=mass_range)

    if not cfg.simulation: # Fix signal shape from MC
        with open(cfg.sim_fit) as f:
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
    yield_signal = zfit.Parameter("yield_signal", len(df_data), 0, len(df_data))
    sig_frac1 = zfit.Parameter("sig_frac1", 0.5, 0, 1)
    # yield_gauss = zfit.Parameter("yield_signal_gauss", len(df_data), 0, len(df_data))

    double_cb = DoubleCB(mean, sigma, alphaL, nL, alphaR, nR, obs=obs)
    # signal_dcb = double_cb.create_extended(yield_signal)
    gauss =zfit.pdf.Gauss(obs=obs, mu=mean, sigma=g_sigma)
    # signal_gauss = gauss.create_extended(yield_gauss)
    model_sig = zfit.pdf.SumPDF(
        [double_cb, gauss], [sig_frac1])
    model_sig_ext = model_sig.create_extended(yield_signal)

    if not cfg.simulation:
        # Background (Exponential)
        lambda_ = zfit.Parameter("lambda", -0.001, -1.0, 0.0)
        yield_bkg = zfit.Parameter("yield_bkg", len(df_data) * 0.2, 0, len(df_data))

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
    ax1.plot(x_plot, total_signal_eval, label=signalname_from_filename(input_files[0]), color="blue", linestyle = "--", linewidth=2)

    if not cfg.simulation:
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

    # Save the results
    params = result.params  # Get the fitted parameters

    # Extract parameter names and values
    fit_results = {param.name: {"value": info["value"], "error": info["hesse"]["error"]} for param, info in params.items()}

    # Save to JSON file
    outname = "mc_res.json" if cfg.simulation else "data_res.json"
    os.makedirs(outputdir, exist_ok=True)

    with open(join(outputdir, outname), "w") as f:
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
    ax2.set_xlabel(r"$ m(B^+)~[\mathrm{MeV}/c^2]$")
    ax1.set_xlim(mass_range[0], mass_range[1])
    ax1.set_ylim(0, 1.1*np.max(counts))

    plt.tight_layout()
    plt.savefig(join(outputdir, "fit_res.png"))
    plt.close()

    # Compute sweights
    if not cfg.simulation:
        weights = compute_sweights(model, masses)

        print(weights)
        signal_weights = weights[yield_signal]
        background_weights = weights[yield_bkg]

        print("signal sWeights: ", signal_weights)
        #print("background sWeights: ", background_weights)

        # df_sw = pd.DataFrame()
        # df_sw["sWeights"] = weights
        df_data["signal_weights"] = signal_weights
        df_data["background_weights"] = background_weights
        # plt.plot(masses, signal_weights, marker=".", linestyle="None", color="red", markersize=0.1, label="signal weights")
        # plt.plot(masses, signal_gauss_weights, marker=".", linestyle="None", color="red", markersize=0.1, label="signal gauss weights")
        plt.plot(masses, df_data["signal_weights"], marker=".", linestyle="None", color="red", markersize=0.1, label="signal")
        plt.plot(masses, df_data["background_weights"], marker=".", linestyle="None", color="green", markersize=0.1, label="background weights")
        plt.plot(masses, background_weights + signal_weights, marker=".", linestyle="None", color="black", markersize=0.1, label="Sum of three")
        plt.xlabel("m($B^{+})~[MeV]/c^{2}$")
        plt.ylabel("weights")
        plt.legend()
        plt.savefig(join(outputdir, "validate_sweights_Bd2JpsiK.png"))
        plt.close()

        tree_dict = {col: np.array(df_data[col]) for col in df_data.columns}
        with uproot.recreate(join(outputdir, "sweights.root")) as f:
            f['DecayTree'] = tree_dict