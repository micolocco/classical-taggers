import DT_utils
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import numpy as np

def plot_used_features(data, features, target_path, log_scale=False, nbins=80):
    """
    Thesis-quality plot of the variables used by the DT,
    grouped by tagging particle class.
    Layout: 2 plots per row (2 columns).
    """

    # ------------------------------
    # Thesis-style global settings
    # ------------------------------
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    })

    # Tagger ordering and your colour scheme
    particles = [
        "OSKaon",
        "OSMuon",
        "OSElectron",
        "SSPion",
        "SSProton",
        "SSKaon",
        "notSamePV",
    ]

    colors = {
        "notSamePV": "green",
        "OSKaon": "violet",
        "OSMuon": "blue",
        "OSElectron": "cyan",
        "SSPion": "orange",
        "SSProton": "brown",
        "SSKaon": "red"
    }

    # ------------------------------
    # Figure layout
    # ------------------------------
    # ------------------------------

    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols    # ceiling division

    # each subplot = (5,4)
    subplot_w, subplot_h = 6, 5

    figwidth = subplot_w * ncols
    figheight = subplot_h * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight))
    axes = axes.flatten()

    plt.subplots_adjust(wspace=0.25, hspace=0.25)


    # ------------------------------
    # Loop over features
    # ------------------------------
    for i, col in enumerate(features):
        ax = axes[i]

        for p in particles:
            vals = data.loc[data["particle"] == p, col]
            ax.hist(
                vals,
                bins=nbins,
                density=True,
                histtype="step",
                lw=2,
                color=colors[p],
                label=p,
               # range=ranges.get(col, None),
            )
        if col == "B_Tr_T_endSV_Z": #set log scale
            #set range on x axis
            ax.set_xlim(0, 200)
        else:
            ax.set_xlim(0, 5)
        #Set x axis label
        ax.set_xlabel(col)

        if log_scale:
            ax.set_yscale('log')  # Set y-axis to logarithmic scale for better visibility

        #    ax.set_yscale('log')
       # ax.set_xlabel(nice_names[col])
        ax.set_ylabel("Probability density")


    # ------------------------------
    # Remove empty axes
    # ------------------------------
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # ------------------------------
    # Shared legend (top center)
    # ------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if log_scale:
        fig.savefig(f"{target_path}/check_SV_log.pdf")
    else:
        fig.savefig(f"{target_path}/check_SV.pdf")
    plt.close(fig)


load_variables= ['B_ENDV_X', 'B_ENDV_Y', 'B_ENDV_Z', 'B_Tr_T_firstX', 'B_Tr_T_firstY', 'B_Tr_T_firstZ', 'B_Tr_T_Origin_Flag', 'B_Tr_T_TRUE_PARTICLE_ID', 'B_Tr_T_MC_MOTHER_ID' ]
tree = "BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree"
f = '/ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/1_raw/Bu2JpsiK/00266989_00000002_1.mc.root'
with uproot.open(f) as f_in:
    df = f_in[tree].arrays(load_variables, library="pd")

df.eval(f'B_Tr_T_absID =abs(B_Tr_T_TRUE_PARTICLE_ID)', inplace = True)

condition_particle_pairs = [
        ((df.B_Tr_T_absID == 321) & (df.B_Tr_T_Origin_Flag == 2), "OSKaon"),
        ((df.B_Tr_T_absID == 13) & (df.B_Tr_T_Origin_Flag == 2), "OSMuon"),
        ((df.B_Tr_T_absID == 11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) != 22), "OSElectron"),
        ((df.B_Tr_T_absID == 211) & (df.B_Tr_T_Origin_Flag == 1), "SSPion"),
        ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1), "SSProton"),
        #((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 2), "OSProton"),
        ((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1), "SSKaon"),
        #((df.B_Tr_T_absID == 321) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag != 100), "otherK"),
        #((df.B_Tr_T_absID == 13) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag != 100), "otherMu"),
        #((df.B_Tr_T_absID == 11) & (df.B_Tr_T_Origin_Flag != 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) != 22) & (df.B_Tr_T_Origin_Flag != 100), "otherE"),
        #((df.B_Tr_T_absID == 11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) == 22), "photonOSEl"),
        #((df.B_Tr_T_absID == 211) & (df.B_Tr_T_Origin_Flag != 1) & (df.B_Tr_T_Origin_Flag != 100), "otherPi"),
        #((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag != 1)  & (df.B_Tr_T_Origin_Flag != 100), "otherP"),
        #((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag != 1) & (df.B_Tr_T_Origin_Flag != 2) & (df.B_Tr_T_Origin_Flag != 100), "noOSSSProton"),
        ((df.B_Tr_T_Origin_Flag == 100), "notSamePV"),
        ] #

    # Separate conditions and particle types for np.select()
conditions = [pair[0] for pair in condition_particle_pairs]
particle_type = [pair[1] for pair in condition_particle_pairs]
# Assign particle types based on conditions, with default "Others" for unmatched rows
df['particle'] = np.select(conditions, particle_type, default="Others")
df = df.loc[(df.particle != 'Others' )]

# Define for X, Y, Z
df.eval(f'B_Tr_T_endSV_X = abs(B_ENDV_X - B_Tr_T_firstX)' , inplace = True)
df.eval(f'B_Tr_T_endSV_Y = abs(B_ENDV_Y - B_Tr_T_firstY)' , inplace = True)
df.eval(f'B_Tr_T_endSV_Z = abs(B_ENDV_Z - B_Tr_T_firstZ)' , inplace = True) 

plot_used_features(data=df, target_path="../", features=[
    'B_Tr_T_endSV_X',
    'B_Tr_T_endSV_Y',
    'B_Tr_T_endSV_Z',
    ], log_scale=False)

plot_used_features(data=df, target_path="../", features=[
    'B_Tr_T_endSV_X',
    'B_Tr_T_endSV_Y',
    'B_Tr_T_endSV_Z',
    ], log_scale=True)