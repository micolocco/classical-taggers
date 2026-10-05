"""Refactored mass-fit driver with optional fixed signal-shape parameters.

This is the parallel replacement for ``mass_fits.py``.  The original driver is
left unchanged.
"""

import argparse
import datetime
import json
import os
from os.path import join
from pprint import pprint
import traceback

from hepstats.splot import compute_sweights
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import pandas as pd
import psutil
import uproot
import zfit

try:
    from scripts.mass_fit.models import Bs2JpsiKstMassModel, GenericMassModel
    from scripts.mass_fit.models.parameters import (
        TAIL_PARAMETER_KEYS,
        load_fixed_signal_parameters,
        signal_parameter_specs,
    )
except ModuleNotFoundError as import_error:
    if import_error.name != "scripts":
        raise
    # Support direct execution as ``python scripts/mass_fit/mass_fit.py``.
    from models import Bs2JpsiKstMassModel, GenericMassModel
    from models.parameters import (
        TAIL_PARAMETER_KEYS,
        load_fixed_signal_parameters,
        signal_parameter_specs,
    )


hep.style.use("LHCb2")

mass_range = None
outputdir = None


def get_tex_decay(decay):
    if decay == "Bu2JpsiK":
        return (
            r"$B^+ \to J/\psi K^+$",
            r"$ m(J/\psi K^{\pm})~[\mathrm{MeV}/c^2]$",
        )
    if decay == "Bd2JpsiKst":
        return (
            r"$B^{0} \to J/\psi K^*$",
            r"$ m(J/\psi K^{*})~[\mathrm{MeV}/c^2]$",
        )
    if decay == "Bs2DsPi":
        return (
            r"$B^{0}_{s} \to D_{s}^{-} \pi^+$",
            r"$ m(D_{s}^{\mp} \pi^{\pm})~[\mathrm{MeV}/c^2]$",
        )
    if decay == "Bs2JpsiKst":
        return (
            r"$B^{0}_{s} \to J/\psi K^*$",
            r"$ m(J/\psi K^{*})~[\mathrm{MeV}/c^2]$",
        )
    raise ValueError(f"Unknown decay type: {decay}")


def fit_valid(result):
    if not result.valid:
        return [False, "fit is not valid, zfit reports invalid fit"]
    if result.edm > 1e-3:
        return [False, "fit is not valid, High edm"]
    relative_uncertainties = [
        info["hesse"]["error"] / abs(info["value"])
        if info["value"] != 0
        else np.nan
        for info in result.params.values()
    ]
    if any(uncertainty > 0.5 for uncertainty in relative_uncertainties):
        return [False, "fit is not valid, High relative uncertainty"]
    return [True, "Fit is valid"]


def plot_mass_fit(
    model_builder,
    masses,
    bins,
    plot_range,
    model,
    params,
    yield_signal,
    yield_bkg,
    destination,
    filename,
    xlabel,
    tex_decay,
    is_simulation,
    yscale="log",
):
    print(f"Generating figures for {tex_decay} fit", flush=True)
    x_plot = np.linspace(plot_range[0], plot_range[1], 1000)
    binwidth = (plot_range[1] - plot_range[0]) / bins
    counts, bin_edges = np.histogram(masses, bins=bins, range=plot_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    errors = np.sqrt(counts)

    _, (ax1, ax2) = plt.subplots(
        2,
        1,
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )
    total_pdf_eval = model_builder.plot_mass_fit(
        ax1,
        x_plot,
        binwidth,
        model,
        params,
        yield_signal,
        yield_bkg,
    )
    ax1.errorbar(
        bin_centers,
        counts,
        yerr=errors,
        fmt="o",
        color="black",
        label="MC Data" if is_simulation else "Data",
        markersize=1,
        elinewidth=1.0,
    )
    ax1.set_ylabel(f"Events$~/~${binwidth}" + r"$[~\mathrm{MeV}/c^2]$")
    ax1.set_yscale(yscale)
    ax1.legend()

    total_fit_at_bin_centers = np.interp(bin_centers, x_plot, total_pdf_eval)
    residuals = np.divide(
        counts - total_fit_at_bin_centers,
        errors,
        out=np.full_like(errors, np.nan, dtype=float),
        where=errors != 0,
    )
    ax2.axhline(0, color="black", linestyle="dashed")
    ax2.axhline(2, color="red", linestyle="dotted")
    ax2.axhline(-2, color="red", linestyle="dotted")
    ax2.bar(
        bin_centers,
        residuals,
        width=binwidth,
        color="gray",
        alpha=0.5,
        label="Residuals",
    )
    ax2.set_ylim(-5, 5)
    ax2.set_ylabel("Pull")
    ax2.set_xlabel(xlabel)
    ax1.set_xlim(plot_range[0], plot_range[1])

    y_min = max(np.min(counts), 2)
    if yscale == "log":
        ax1.set_ylim(y_min * 0.66, np.max(counts) * 2)
    else:
        ax1.set_ylim(0, np.max(counts) * 1.2)
    plt.tight_layout()
    plt.savefig(join(destination, filename))
    print(f"Figure saved to {join(destination, filename)}", flush=True)
    plt.close()

    differences = counts - total_fit_at_bin_centers
    plt.plot(bin_centers, differences)
    plt.xlabel(xlabel)
    plt.ylabel("Absolute Fit Error")
    plt.tight_layout()
    plt.savefig(join(destination, filename.replace(".pdf", "_Abs_Fit_error.png")))
    plt.close()

    relative_differences = np.divide(
        differences,
        counts,
        out=np.full_like(differences, np.nan, dtype=float),
        where=counts != 0,
    )
    plt.plot(bin_centers, relative_differences)
    plt.xlabel(xlabel)
    plt.ylabel("Relative Fit Error")
    plt.tight_layout()
    plt.savefig(join(destination, filename.replace(".pdf", "_Rel_Fit_error.png")))
    plt.close()


def _shape_parameters_for_fit(decay, simulation, sim_fit, fixed_shape_params):
    if fixed_shape_params:
        if simulation or not sim_fit:
            parameters = load_fixed_signal_parameters(fixed_shape_params, decay)
            print(
                "Loaded and fixed all signal-shape parameters from "
                f"{fixed_shape_params}",
                flush=True,
            )
            return parameters

        non_tail_keys = tuple(
            key
            for key in signal_parameter_specs(decay)
            if key not in TAIL_PARAMETER_KEYS
        )
        parameters = load_fixed_signal_parameters(
            fixed_shape_params,
            decay,
            keys=non_tail_keys,
        )
        parameters.update(
            load_fixed_signal_parameters(
                sim_fit,
                decay,
                keys=TAIL_PARAMETER_KEYS,
                require_error=False,
            )
        )
        print(
            "Loaded and fixed non-tail signal-shape parameters from "
            f"{fixed_shape_params} and MC tail parameters from {sim_fit}",
            flush=True,
        )
        return parameters
    if simulation:
        return None
    if not sim_fit:
        raise ValueError(
            "--sim_fit is required for data fits unless --fixed_shape_params is given"
        )
    parameters = load_fixed_signal_parameters(
        sim_fit,
        decay,
        keys=TAIL_PARAMETER_KEYS,
        require_error=False,
    )
    print(f"Loaded fixed MC tail parameters from {sim_fit}", flush=True)
    return parameters


def massfit(
    obs,
    masses,
    tex_decay,
    xlabel,
    outname,
    simulation,
    sim_fit,
    filename,
    df,
    compute_weights,
    generate_figures,
    obs_name,
    prefix="",
    is_selected=True,
    bins=100,
    fixed_shape_params=None,
    decay_type=None,
):
    if mass_range is None or outputdir is None:
        raise RuntimeError("mass_range and outputdir must be configured before massfit")
    if decay_type is None:
        decay_type = _decay_type_from_tex(tex_decay)
    print(f"Starting mass fit for {tex_decay}", flush=True)
    shape_parameters = _shape_parameters_for_fit(
        decay_type,
        simulation,
        sim_fit,
        fixed_shape_params,
    )
    model_builder = GenericMassModel.from_decay_type(
        obs,
        tex_decay,
        simulation,
        is_selected,
        len(df),
        shape_parameters,
    )
    fit_context = model_builder.build()
    model = fit_context["model"]
    yield_signal = fit_context["yield_signal"]
    yield_bkg = None if simulation else fit_context["yield_bkg"]

    data = zfit.Data.from_numpy(obs=obs, array=masses)
    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit(gradient=False, tol=1e-4, verbosity=5)
    result = minimizer.minimize(nll)
    result.hesse()
    print(
        f"Initial fit converged: {result.converged} with edm: {result.edm}",
        flush=True,
    )
    validity = fit_valid(result)
    print(f"Initial fit\n{validity[1]}", flush=True)
    print(result, flush=True)
    attempt = 0
    while not validity[0] and attempt < 3:
        result = minimizer.minimize(nll, init=result)
        result.hesse()
        validity = fit_valid(result)
        print(f"Fit attempt {attempt + 1}\n{validity[1]}", flush=True)
        print(result, flush=True)
        attempt += 1
    if not validity[0]:
        result = minimizer.minimize(nll, init=result)
        result.hesse()

    print(f"correlation matrix:\n{result.correlation()}", flush=True)
    print(f"Result message: {result.message}", flush=True)
    params = result.params
    try:
        fit_results = {
            parameter.name: {
                "value": info["value"],
                "error": info["hesse"]["error"],
            }
            for parameter, info in params.items()
        }
        os.makedirs(outputdir, exist_ok=True)
        with open(join(outputdir, f"{outname}.json"), "w", encoding="utf-8") as output:
            json.dump(fit_results, output, indent=4)
    except Exception:
        print("Error while saving fit results to JSON:")
        traceback.print_exc()

    if generate_figures:
        plot_mass_fit(
            model_builder,
            masses,
            bins,
            mass_range,
            model,
            params,
            yield_signal,
            yield_bkg,
            outputdir,
            prefix + filename,
            xlabel,
            tex_decay,
            simulation,
        )
        plot_mass_fit(
            model_builder,
            masses,
            bins,
            mass_range,
            model,
            params,
            yield_signal,
            yield_bkg,
            outputdir,
            prefix + "linear_" + filename,
            xlabel,
            tex_decay,
            simulation,
            yscale="linear",
        )

    if compute_weights:
        if isinstance(model_builder, Bs2JpsiKstMassModel) and not simulation:
            lower_bound = 5300
            bins = int(
                bins
                * (mass_range[1] - lower_bound)
                / (mass_range[1] - mass_range[0])
            )
            df = df.loc[df[obs_name] > lower_bound]
            masses = df[obs_name].values
            bs_context = model_builder.build_bs_region_context(
                fit_context["signal_params"],
                yield_signal,
                yield_bkg,
                mass_range,
                lower_bound=lower_bound,
            )
            model = bs_context["model"]
            obs_restricted = bs_context["obs"]
            data = zfit.Data.from_numpy(obs=obs_restricted, array=masses)
            nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
            minimizer = zfit.minimize.Minuit(
                gradient=False,
                tol=1e-4,
                verbosity=5,
            )
            result = minimizer.minimize(nll)
            result.hesse()
            params = result.params
            print(result)
            plot_mass_fit(
                model_builder,
                masses,
                bins,
                (lower_bound, mass_range[1]),
                model,
                params,
                yield_signal,
                yield_bkg,
                outputdir,
                "Bs_region_" + prefix + filename,
                xlabel,
                tex_decay,
                simulation,
            )

        print("computing SWeights", flush=True)
        print(masses.shape)
        weights = compute_sweights(model, masses)
        print(weights)
        df[f"{prefix}signal_weights"] = weights[yield_signal]
        for key, value in weights.items():
            if key != yield_signal:
                df[f"{prefix}{key}_weights"] = value

        if generate_figures:
            weight_sum = np.zeros_like(masses)
            for key, value in weights.items():
                plt.plot(
                    masses,
                    value,
                    marker=".",
                    linestyle="None",
                    markersize=1,
                    label=key,
                )
                weight_sum += value
            plt.plot(
                masses,
                weight_sum,
                marker=".",
                linestyle="None",
                color="black",
                markersize=1,
                label="Sum of all weights",
            )
            handles = [
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    label=str(key).split("=")[0],
                    color=plt.gca().lines[index].get_color(),
                )
                for index, key in enumerate(weights)
            ]
            plt.legend(handles=handles)
            plt.grid(True, alpha=0.3)
            plt.xlabel(xlabel)
            plt.savefig(join(outputdir, prefix + "validate_sweights.png"))
            plt.close()
        del weights

    print(f"Fit ended with status: {result.status}", flush=True)
    return df


def _decay_type_from_tex(tex_decay):
    for decay in ("Bu2JpsiK", "Bd2JpsiKst", "Bs2DsPi", "Bs2JpsiKst"):
        if get_tex_decay(decay)[0] == tex_decay:
            return decay
    raise ValueError(f"Unknown decay label: {tex_decay}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mass fits for B decays to extract sWeights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_files", nargs="+")
    parser.add_argument(
        "--treename",
        help="TreeName of the input file",
        default="BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree",
    )
    parser.add_argument("--range", help="Observable range", nargs="+")
    parser.add_argument("--output", help="Where fit results and plots will be stored")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument(
        "--sim_fit",
        help="MC fit-result JSON used to fix tail parameters in data fits",
    )
    parser.add_argument(
        "--fixed_shape_params",
        dest="fixed_shape_params",
        help=(
            "Fit-result JSON used to fix signal-shape parameters. In data fits, "
            "non-tail parameters are read from this file and tail parameters from "
            "--sim_fit. Each entry must have the form "
            "{'value': number, 'error': number}."
        ),
    )
    parser.add_argument("--decay_type", help="Decay used")
    parser.add_argument(
        "--obs_name",
        help="Name of observable",
        default="B_DTF_PV_Jpsi_MASS",
    )
    parser.add_argument(
        "--num_threads",
        help="Number of threads to use for the fits",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--selected",
        action="store_true",
        help="Whether the input data has already been selected with the BDT cut.",
    )
    return parser.parse_args()


def main():
    global mass_range, outputdir

    cfg = parse_args()
    pprint(cfg)
    massname = cfg.obs_name
    mass_range = (int(cfg.range[0]), int(cfg.range[1]))
    outputdir = cfg.output
    os.makedirs(outputdir, exist_ok=True)
    zfit.run.set_n_cpu(n_cpu=cfg.num_threads)

    print(f"Reading files started on {datetime.datetime.now().strftime('%H:%M:%S')}")
    variables = [
        "file_id",
        "RUNNUMBER",
        "EVENTNUMBER",
        massname,
        "candidate_index",
    ]
    if cfg.simulation:
        variables.append("B_BKGCAT")

    df_data = None
    for index, input_file in enumerate(cfg.input_files):
        print(f"Reading input file {index + 1}/{len(cfg.input_files)}: {input_file}")
        memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        print(f"Megabites used: {memory_mb}", flush=True)
        with uproot.open(input_file) as root_file:
            current_df = root_file[cfg.treename].arrays(variables, library="pd")
        current_df.dropna(inplace=True)
        if cfg.simulation:
            background_category = 20 if cfg.decay_type == "Bs2DsPi" else 0
            current_df = current_df.query(f"B_BKGCAT == {background_category}").copy()
        current_df["event_entry"] = (
            current_df["file_id"].astype(str)
            + "_"
            + current_df["RUNNUMBER"].astype(str)
            + "_"
            + current_df["EVENTNUMBER"].astype(str)
        )
        current_df = current_df.groupby("event_entry").first().reset_index()
        print(
            "File read. Number of events: "
            f"{len(current_df['event_entry'].unique())}, "
            f"number of candidates: {len(current_df)}",
            flush=True,
        )
        df_data = (
            current_df
            if df_data is None
            else pd.concat([df_data, current_df], ignore_index=True)
        )
    print(f"Reading files ended on {datetime.datetime.now().strftime('%H:%M:%S')}")

    df_data = df_data.query(
        f"{massname} < {mass_range[1]} and {massname} > {mass_range[0]}"
    )
    masses = df_data[massname].values
    obs = zfit.Space("mass", limits=mass_range)
    tex_decay, xlabel = get_tex_decay(cfg.decay_type)
    selected_string = "selected" if cfg.selected else "non_selected"
    prefix = "" if cfg.selected else "non_selected_"
    df_data = massfit(
        obs,
        masses,
        tex_decay,
        xlabel,
        f"event_{selected_string}_fit",
        cfg.simulation,
        cfg.sim_fit,
        f"fit_res_{selected_string}.pdf",
        df_data,
        compute_weights=not cfg.simulation,
        generate_figures=True,
        obs_name=cfg.obs_name,
        is_selected=cfg.selected,
        prefix=prefix,
        fixed_shape_params=cfg.fixed_shape_params,
        decay_type=cfg.decay_type,
    )

    tree_dict = {
        column: np.array(df_data[column])
        for column in df_data.columns
        if column not in {"event_entry"}
    }
    print(tree_dict)
    for column, values in tree_dict.items():
        print(f"{column}: {values[0]} ({type(values[0])})")
    del df_data
    print("Writing file to disk")
    with uproot.recreate(join(outputdir, f"weights_{selected_string}.root")) as output:
        output["DecayTree"] = tree_dict
    print(f"Mass fit script ended on {datetime.datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
