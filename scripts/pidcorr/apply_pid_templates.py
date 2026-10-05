import json
from typing import OrderedDict

import numpy as np 
import pandas as pd
from tqdm import tqdm
import uproot
import matplotlib.pyplot as plt
from pidgen2.tools.plot import set_lhcb_style

import pickle
from pidgen2.tools.template import TemplateManager
from pidgen2.tools.correction import Corrector
from pidgen2.tools.datasets import DatasetManager
from pidgen2.tools.transform import probnn_transformer
set_lhcb_style()

#Constants used when pickling the templated
bins   = [70, 70, 70, 20]
all_ranges = {
    "PROBNN_PI" : [(  0,  1), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    "PROBNN_K"  : [(  0,  1), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    "PROBNN_P"  : [(  0,  1), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    "PROBNN_MU" : [(  0,  1), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    "PROBNN_E"  : [(  0,  1), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    # "PID_PI"    : [(-90, 50), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    "PID_K"     : [(-90, 50), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    "PID_P"     : [(-90, 50), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    "PID_MU"    : [(-15, 10), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
    "PID_E"     : [(-15, 10), (-5, 4), (1.8, 5.2), ( 0., 3.5)],
}
sigma  = [2., 4., 4., 4.]


#Id by True Particle label
id_by_true_particle = {
    "pi" : 211,
    "K"  : 321,
    "P"  : 2212,
    "Mu" : 13,
    "E"  : 11
}

# TODO 
#- add all of the arguments to make it opperable from snakemake



def load_all_correctors(template_paths_by_TrueID_and_Probe):
    correctors = OrderedDict()
    for true_id, probes in template_paths_by_TrueID_and_Probe.items():
        correctors[true_id] = {}
        for probe, paths in probes.items():
            print(f"Loading templates for true_id: {true_id}, probe: {probe}", flush=True)

            calib_path, mc_path = paths
            with open(calib_path, "rb") as f:
                tm = pickle.load(f)
            values, variances, edges = tm.get_template()

            with open(mc_path, "rb") as f:
                tm_mc = pickle.load(f)
            mc_values, mc_variances, _ = tm_mc.get_template()

            corrector = Corrector(
                mc_values, mc_variances,
                values, variances,
                edges,
                transformer=probnn_transformer,
                nan=-1.,
                verbosity=2,
            )
            corrector.add_cdf("nominal", sigma=sigma, min_counts=10.)


            correctors[true_id][probe] = corrector
    return correctors

def get_original_column_name(var_name):
    if var_name.startswith("PID"):
        probe = var_name.split("PID")[1]

        if probe == "E" or probe == "MU":
            probe = probe.lower()

        return f"B_Tr_T_PID{probe}"
    elif var_name.startswith("PROBNN"):
        probe = var_name.split("_")[1]
        return f"B_Tr_T_{var_name}"
    else:
        raise ValueError(f"Unknown variable name: {var_name}")

def apply_correction(chunk, correctors):
    corrected_columns = []
    original_columns = []

    for true_id, probes in correctors.items():
        for var_name, corrector in probes.items():
            original_name =  get_original_column_name(var_name)

            corrected_name = original_name + "_corr"

            branches_used_tuples = {
                "pid":  original_name,
                "pt" : "B_Tr_T_PT",
                "eta": "B_Tr_T_Eta",
                "ntr": "B_nTracks",
            }


            chunk[corrected_name] = chunk[original_name].copy()  # Initialize with original values

            # Correct only for rows where the true particle ID matches
            mask = chunk['B_Tr_T_absID'] == id_by_true_particle[true_id]


            corrected_pid = corrector.correct(data=chunk[mask], expr=branches_used_tuples, name="nominal").astype(chunk[original_name].dtype)  # Ensure the dtype matches the original column

            corrected_pid = corrected_pid
            chunk.loc[mask, corrected_name] = corrected_pid

            if corrected_name not in corrected_columns:
                corrected_columns.append(corrected_name)
                original_columns.append(original_name)
    return chunk[corrected_columns + original_columns]



dm = DatasetManager()
in_path = "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bd2JpsiKst/00267659_00000002_1.mc.root"
outpath = "/ceph/users/togasa/pidcorr/" + in_path.split("/")[-1]
block = 'block1'
tree = "DecayTree;1"
batch_size = 1_000_000

template_paths=[
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_E_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_E_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_MU_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_MU_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_PI_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_PI_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_K_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_K_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_P_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/pi_PROBNN_P_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_E_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_E_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_MU_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_MU_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_PI_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_PI_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_K_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_K_mc.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_P_calib.pkl',
    '/ceph/users/togasa/pidcorr/block1/K_PROBNN_P_mc.pkl',

    
    #PID templates are broken at the minute, probably a binning issue. So we skip them for now.
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_E_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_E_mc.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_MU_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_MU_mc.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_K_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_K_mc.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_P_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_P_mc.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_E_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_E_mc.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_MU_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_MU_mc.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_K_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_K_mc.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_P_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_P_mc.pkl',

    # '/ceph/users/togasa/pidcorr/block1/K_PID_PI_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/K_PID_PI_mc.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_PI_calib.pkl',
    # '/ceph/users/togasa/pidcorr/block1/pi_PID_PI_mc.pkl',
]


template_paths_by_TrueID_and_Probe = {}
for path in template_paths:
    if path.endswith("calib.pkl"):
        true_id  = path.split("/")[-1].split("_")[0]
        var_type = path.split("/")[-1].split("_")[1]
        probe    = path.split("/")[-1].split("_")[2]

        if probe == 'PI' and var_type == 'PID': # No PIDPi in the datasets used. PID is defined relative to Pions. No idea what PIDPi is even supposed to be. So we skip it for now.
            continue

        mc_path = path.replace("calib.pkl", "mc.pkl")

        if var_type == 'PID': #Remove _ for PID variables to match the variable names in the ntuples
            var_name = f"{var_type}{probe}"
        else:
            var_name = f"{var_type}_{probe}"

        if mc_path not in template_paths:
            raise ValueError(f"MC template for {true_id} and {probe} not found: {mc_path}")

        if true_id not in template_paths_by_TrueID_and_Probe:
            template_paths_by_TrueID_and_Probe[true_id] = {}
        template_paths_by_TrueID_and_Probe[true_id][var_name] = [path, mc_path]

print(json.dumps(template_paths_by_TrueID_and_Probe, indent=4))


correctors = load_all_correctors(template_paths_by_TrueID_and_Probe)

in_paths = [
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000001_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000002_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000003_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000004_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000005_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000006_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000007_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000008_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000009_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000010_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000011_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000012_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000013_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000014_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000015_1.mc.root",
]

in_paths = [
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bs2DsPi/00266991_00000001_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bs2DsPi/00266991_00000002_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bs2DsPi/00266991_00000003_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bs2DsPi/00266991_00000004_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bs2DsPi/00266991_00000005_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bs2DsPi/00266991_00000006_1.mc.root",
]




for in_path in tqdm(in_paths, desc="Processing files"):
    outpath = "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bs2DsPi/pidcorr/" + in_path.split("/")[-1]

    pid_df = pd.DataFrame()

    with uproot.recreate(outpath) as fout:
        chunk_iter = uproot.iterate({in_path: tree}, library="pd", step_size=batch_size)

        for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
            _df = apply_correction(chunk, correctors)

            

            if i == 0:
                fout["DecayTree"] = chunk
                pid_df = _df
            else:
                fout["DecayTree"].extend(chunk)
                pid_df = pd.concat([pid_df, _df], ignore_index=True)

    plot_vars = [i for i in pid_df.columns if not i.endswith("_corr")]

    print(f"Plotting histograms for the following variables: {plot_vars}", flush=True)

    for orig_column in plot_vars:
        print(f"Plotting histograms for {orig_column}", flush=True)
        corrected_column = f"{orig_column}_corr"


        sorted_combined = np.sort(np.concatenate([pid_df[orig_column], pid_df[corrected_column]]))
        min_val = sorted_combined[int(0.01 * len(sorted_combined))]
        max_val = sorted_combined[int(0.99 * len(sorted_combined))]
        bins = np.linspace(min_val, max_val, 101)

        mask = pid_df[orig_column].to_numpy() != pid_df[corrected_column].to_numpy()  # Mask for rows where the values differ

        print(f"Fraction of tracks with correction applied for {orig_column}: {np.sum(mask) / len(mask):.4%}", flush=True)
        print(f"Fraction of non -1 corrected values for {orig_column}: {np.sum(pid_df[corrected_column] != -1) / len(pid_df):.4%}", flush=True)
        print(f"Correction efficiency for {orig_column}: {np.sum((pid_df[corrected_column] != -1) & mask) / np.sum(mask):.4%}", flush=True)
        
        plt.figure(figsize=(10, 6))
        plt.hist(pid_df      [orig_column     ], bins=bins, range=(min_val, max_val), histtype="step", label="Uncorrected", color="blue", alpha=0.7)
        plt.hist(pid_df      [corrected_column], bins=bins, range=(min_val, max_val), histtype="step", label="Corrected", color="orange", alpha=0.7)
        plt.hist(pid_df[mask][orig_column     ], bins=bins, range=(min_val, max_val), histtype="step", label="Uncorrected (changed)", color="blue", alpha=0.7, linestyle='dashed')
        plt.hist(pid_df[mask][corrected_column], bins=bins, range=(min_val, max_val), histtype="step", label="Corrected (changed)", color="orange", alpha=0.7, linestyle='dashed')


        plt.legend()
        plt.xlabel(orig_column)
        plt.ylabel("Counts")
        plt.savefig(f"{orig_column}_correction_hist.pdf", dpi=150)

        plt.yscale("log")
        plt.ylabel("Counts")
        plt.savefig(f"log_{orig_column}_correction_hist.pdf", dpi=150)
        plt.clf()  # Clear the figure for the next plot


