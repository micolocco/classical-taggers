#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scripts import matplotlib_lhcb_style

# Apply LHCb style
matplotlib_lhcb_style(plt)

# ================== CONFIG ==================

#DATA_BU_DIR = "/ceph/users/molocco/FlavourTagging/data/withUT_MC_2024/2_added_features/Bu2JpsiK/"
#DATA_BD_DIR = "/ceph/users/molocco/FlavourTagging/data/withUT_MC_2024/2_added_features/Bd2JpsiKst/"
#MC_BU_DIR   = "/ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/2_added_features/Bu2JpsiK/"
#MC_BD_DIR   = "/ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/2_added_features/Bd2JpsiKst/"
#TREE = "Tuple/DecayTree"
#selected
MC_BU_DIR   = "/ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/3_selected/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK/balanced/union_PROBNN/"
MC_BD_DIR   = "/ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/3_selected/Bd2JpsiKst/OSKaon/allBKGCAT_notSamePV_noOSP_SSK/balanced/union_PROBNN/"
DATA_BD_DIR = "/ceph/users/molocco/FlavourTagging/data/withUT_MC_2024/3_selected/Bd2JpsiKst/OSKaon/allBKGCAT_notSamePV_noOSP_SSK/balanced/union_PROBNN/"
DATA_BU_DIR = "/ceph/users/molocco/FlavourTagging/data/withUT_MC_2024/3_selected/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK/balanced/union_PROBNN/"
TREE = "DecayTree"



# Variables we may plot (union of everything we care about)
VARS = [
    "B_nPVs",
    "B_nTracks",
    "B_Tr_T_P",
    "B_Tr_T_PT",
    "B_PT",
    "B_Tr_T_OWNPVIPSig",
    "B_Tr_T_CHI2DOF",
    "B_Tr_T_PROBNN_E",
    "B_Tr_T_PROBNN_K",
    "B_Tr_T_PROBNN_P",
    "B_Tr_T_PROBNN_MU",
    "B_Tr_T_PROBNN_PI",
    "B_Tr_T_PROBNN_GHOST",
    "B_Tr_T_absOWNPV_IP",
    "B_Tr_T_DeltaR",
    "RUNNUMBER",
    "EVENTNUMBER",
]

OUTDIR = Path("plots_features")

LBL = {
    "Bu_data": r"$B^+\!\to\!J/\psi K^+$ (data)",
    "Bu_mc":   r"$B^+\!\to\!J/\psi K^+$ (MC)",
    "Bd_data": r"$B^0\!\to\!J/\psi K^{*0}$ (data)",
    "Bd_mc":   r"$B^0\!\to\!J/\psi K^{*0}$ (MC)",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
})

# --- Feature config ---
# kind: "track" -> all tracks; "event" -> one entry per event
# integer: True -> integer bins
FEATURES = {
    # event-level
    "B_nPVs": {
        "kind": "event",
        "integer": True,
        range: (0, 15),
        "xlabel": "Number of primary vertices",
    },
    "B_nTracks": {
        "kind": "event",
        "integer": False,
        "range": (0, 500),
        "xlabel": "Number of tracks",
    },
    # track-level PROBNNs
    "B_Tr_T_PROBNN_E": {
        "kind": "track",
        "integer": False,
        "xlabel": r"$\mathrm{ProbNN}_e$",
        "range": (0.0, 1.0),
        "nbins": 50,
    },
    "B_Tr_T_PROBNN_K": {
        "kind": "track",
        "integer": False,
        "xlabel": r"$\mathrm{ProbNN}_K$",
        "range": (0.0, 1.0),
        "nbins": 50,
    },
    "B_Tr_T_PROBNN_P": {
        "kind": "track",
        "integer": False,
        "xlabel": r"$\mathrm{ProbNN}_p$",
        "range": (0.0, 1.0),
        "nbins": 50,
    },
    "B_Tr_T_PROBNN_MU": {
        "kind": "track",
        "integer": False,
        "xlabel": r"$\mathrm{ProbNN}_\mu$",
        "range": (0.0, 1.0),
        "nbins": 50,
    },
    "B_Tr_T_PROBNN_PI": {
        "kind": "track",
        "integer": False,
        "xlabel": r"$\mathrm{ProbNN}_\pi$",
        "range": (0.0, 1.0),
        "nbins": 50,
    },
    "B_Tr_T_PROBNN_GHOST": {
        "kind": "track",
        "integer": False,
        "xlabel": r"$\mathrm{ProbNN}_\mathrm{ghost}$",
        "range": (0.0, 1.0),
        "nbins": 50,
    },
    ## you can add more features (P, PT, DeltaR, etc.) here later
}

# ================== HELPERS ==================

def load_dataframe(directory: str) -> pd.DataFrame:
    """Track-level DataFrame from all ROOT files in a directory."""
    files = sorted(glob.glob(str(Path(directory) / "*.root")))
    if not files:
        raise FileNotFoundError(f"No ROOT files found in {directory}")
    dfs = []
    for f in files:
        with uproot.open(f) as f_in:
            df = f_in[TREE].arrays(VARS, library="pd")
            df.dropna(inplace=True)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def common_bins_integer(*arrays: pd.Series) -> np.ndarray:
    a = np.concatenate([s.to_numpy() for s in arrays if len(s)])
    lo = int(np.floor(a.min()))
    hi = int(np.ceil(a.max()))
    return np.arange(lo - 0.5, hi + 1.5, 1.0)


def common_bins_linear(*arrays: pd.Series, nbins: int = 60, qhi: float = 0.995) -> np.ndarray:
    a = np.concatenate([s.to_numpy() for s in arrays if len(s)])
    lo = float(np.nanmin(a))
    hi = float(np.nanquantile(a, qhi))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, nbins + 1)


def make_bins(arrays, cfg):
    if cfg.get("integer", False):
        return common_bins_integer(*arrays)
    if "range" in cfg:
        lo, hi = cfg["range"]
        nbins = cfg.get("nbins", 50)
        return np.linspace(lo, hi, nbins + 1)
    nbins = cfg.get("nbins", 60)
    qhi = cfg.get("qhi", 0.995)
    return common_bins_linear(*arrays, nbins=nbins, qhi=qhi)


def plot_feature(feat_name, cfg, dfs, kind_label):
    """Plot single feature for Bu/Bd, data/MC."""
    # dfs is a dict with keys: "Bu_data", "Bu_mc", "Bd_data", "Bd_mc"
    arrays_for_bins = [
        dfs["Bu_data"][feat_name],
        dfs["Bu_mc"][feat_name],
        dfs["Bd_data"][feat_name],
        dfs["Bd_mc"][feat_name],
    ]
    bins = make_bins(arrays_for_bins, cfg)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Bu: blue
    ax.hist(dfs["Bu_data"][feat_name], bins=bins, density=True,
            histtype="step", color="C0", linestyle="-", label=LBL["Bu_data"])
    ax.hist(dfs["Bu_mc"][feat_name],   bins=bins, density=True,
            histtype="step", color="C0", linestyle="--", label=LBL["Bu_mc"])

    # Bd: orange
    ax.hist(dfs["Bd_data"][feat_name], bins=bins, density=True,
            histtype="step", color="C1", linestyle="-", label=LBL["Bd_data"])
    ax.hist(dfs["Bd_mc"][feat_name],   bins=bins, density=True,
            histtype="step", color="C1", linestyle="--", label=LBL["Bd_mc"])

    ax.set_xlabel(cfg.get("xlabel", feat_name))
    ax.set_ylabel("Normalized counts")
    if feat_name not in ["B_nPVs", "B_nTracks"]:
        ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.tight_layout()

    OUTDIR.mkdir(exist_ok=True)
    safe_name = feat_name.replace("B_Tr_T_", "").replace("B_", "")
    outbase = f"{safe_name}_data_vs_MC_{kind_label}"
    #for ext in ("pdf", "png"):
    for ext in ("png",):
        fig.savefig(OUTDIR / f"{outbase}_selected_OSKaon.{ext}", dpi=300)
    plt.close(fig)


# ================== MAIN ==================

def main():
    print("Loading track-level data and MC...")
    dfs_track = {
        "Bu_data": load_dataframe(DATA_BU_DIR),
        "Bd_data": load_dataframe(DATA_BD_DIR),
        "Bu_mc":   load_dataframe(MC_BU_DIR),
        "Bd_mc":   load_dataframe(MC_BD_DIR),
    }

    # ---------- 1) TRACK-LEVEL FEATURES ----------
    print("Plotting track-level features (e.g. PROBNNs)...")
    for feat, cfg in FEATURES.items():
        if cfg["kind"] != "track":
            continue
        print(f"  - {feat} (track)")
        plot_feature(feat, cfg, dfs_track, kind_label="track")

    # ---------- 2) SHRINK TO EVENT-LEVEL ----------
    print("Converting DataFrames to event level...")
    for key in dfs_track:
        dfs_track[key] = dfs_track[key].drop_duplicates(
            subset=["RUNNUMBER", "EVENTNUMBER"]
        )
    # ---------- 3) EVENT-LEVEL FEATURES ----------
    print("Plotting event-level features (e.g. nPVs, nTracks)...")
    for feat, cfg in FEATURES.items():
        if cfg["kind"] != "event":
            continue
        print(f"  - {feat} (event)")
        plot_feature(feat, cfg, dfs_track, kind_label="event")

    print(f"Plots saved in: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
