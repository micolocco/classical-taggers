import pandas as pd
import uproot
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt

def read_files(files, vars, treename):
    df = pd.DataFrame(columns=vars)
    vars.remove('event_entry')

    for i, f in enumerate(files):
        print(f"Reading input file {i+1}/{len(files)}: {f}", flush=True)
        print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB')

        id = os.path.basename(f)[:-5]
        if id[-7:-2] == '.data':
            id = id[:-7]
        else:
            id = id[:-3]

        with uproot.open("{}".format(f)) as _f:
            _df = _f[treename].arrays(vars, library="pd")
        _df.dropna(inplace = True)
        _df["event_entry"] = id + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER'], inplace=True)
        print(_df.shape)
        if i == 0:
            df = _df.copy()
        else:
            df = pd.concat([df, _df.copy()], ignore_index = True)
        print(df.shape)
    return df

outpath = '/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/1_event_selected/controlplots' #For no particular reason
path = '/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/2_added_features/Bu2JpsiK'
path = '/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/4_weighted/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/'
#get all files in the directory
files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.root')][:50]
print(files)

#get the column names of the first file
with uproot.open(files[0]) as f:
    tree = f["DecayTree;1"]
    print(tree.keys())
    vars = tree.keys()
    vars.append('event_entry')

df = read_files(files, vars, "DecayTree;1")

BID = 'B_ID' # 'B_TRUEID'

#make a plot of all columns in the dataframe split by BID

cols = [col for col in df.columns if col != BID and df[col].dtype != 'O']
n_cols = 3
n_rows = int(np.ceil(len(cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    for bid_value in df[BID].unique():
        data = df[df[BID] == bid_value][col]
        axes[i].hist(data, bins=50, alpha=0.5, label=f"{BID}={bid_value}")
        axes[i].set_yscale('log')
    axes[i].set_title(col)
    axes[i].legend()
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("control_plots.png", dpi=300)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    bins = np.linspace(df[col].min(), df[col].max(), 50)
    for bid_value in df[BID].unique():
        data = df[df[BID] == bid_value][col]
        axes[i].hist(data, bins=bins, alpha=0.5, label=f"{BID}={bid_value}", weights=df['signal_weights'][df[BID] == bid_value])
        axes[i].set_yscale('log')
    axes[i].set_title(col)
    axes[i].legend()
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("control_plots_sweighted.png", dpi=300)