import uproot
import pandas as pd
import os
from functools import reduce
import lhcb_ftcalib as ft
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import numpy as np
from matplotlib.ticker import FixedLocator


def load_files(paths, taggers):
    files = [f for f in os.listdir(paths[next(iter(paths))]) if f.endswith('.root')]
    df = None
    for f in files:

        file_dfs = []
        for tagger in taggers:
        
            for prefix, path in paths.items():
                path_tagger = path.replace('OSKaon', tagger)
                file = join(path_tagger, f)
                print(f'Processing {file}')

                id = f[:-5]
                if id[-7:-2] == '.data':
                    id = id[:-7]
                else:
                    id = id[:-3]

                with uproot.open(file) as _f:
                    _df = _f["DecayTree"].arrays([f'{tagger}_TagDec',f'{tagger}_Eta', 'RUNNUMBER', 'EVENTNUMBER', 'B_ID'], library="pd")
                _df.dropna(inplace = True)


                _df["event_entry"] = id + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
                _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER'], inplace=True)

                _df.rename(columns={f'{tagger}_TagDec': f'{prefix}_{tagger}_TagDec', f'{tagger}_Eta': f'{prefix}_{tagger}_Eta'}, inplace=True)
                file_dfs.append(_df)


        
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=['event_entry', 'B_ID'], how="outer"), file_dfs)



        if df is None:
            df = merged_df
        else:
            df = pd.concat([df, merged_df], ignore_index=True)

    for prefix in paths.keys():
        f_taggers = ft.TaggerCollection()

        for tagger in taggers:
            print(f'creating {tagger} for {prefix}')
            f_taggers.create_tagger(f"{tagger}", eta_data =df[f'{prefix}_{tagger}_Eta'].tolist(), dec_data = df[f'{prefix}_{tagger}_TagDec'].tolist(), B_ID =df['B_ID'].tolist(), mode = 'Bu', )

        tagger_combination = f_taggers.combine_taggers(f'{prefix}OSComb', calibrated=False)

        print(f'tagger_combination: {f_taggers["OSKaon"].stats.tagging_power(calibrated=False)}')
        print(f'tagger_combination: {tagger_combination.stats.tagging_power(calibrated=False)}')

        df[f'{prefix}_OSComb_Eta'] = tagger_combination.stats._full_data.eta
        print(df[f'{prefix}_OSComb_Eta'])
    
    return df


#Correlation of this studies models to reference models                
paths = {
    'ref' : "/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024_micols/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_MC/",
    'data' : "/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_Data/",
    'mc' : "/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_MC/",
    }

taggers = ['OSKaon', 'OSMuon', 'OSElectron']

df = load_files(paths, taggers)

column_order = ['ref_OSKaon_Eta' , 'ref_OSMuon_Eta' , 'ref_OSElectron_Eta' , 'ref_OSComb_Eta' , 'seperator1',
                'mc_OSKaon_Eta'  , 'mc_OSMuon_Eta'  , 'mc_OSElectron_Eta'  , 'mc_OSComb_Eta'  , 'seperator2',
                'data_OSKaon_Eta', 'data_OSMuon_Eta', 'data_OSElectron_Eta', 'data_OSComb_Eta', 'seperator3',
                ]


translation_dict = {
    'ref_OSKaon_Eta': 'OSKaon reference',
    'ref_OSMuon_Eta': 'OSMuon reference',
    'ref_OSElectron_Eta': 'OSElectron reference',
    'ref_OSComb_Eta': 'OSCombination reference',
    'mc_OSKaon_Eta': 'OSKaon simulation',
    'mc_OSMuon_Eta': 'OSMuon simulation',
    'mc_OSElectron_Eta': 'OSElectron simulation',
    'mc_OSComb_Eta': 'OSCombination simulation',
    'data_OSKaon_Eta': 'OSKaon data',
    'data_OSMuon_Eta': 'OSMuon data',
    'data_OSElectron_Eta': 'OSElectron data',
    'data_OSComb_Eta': 'OSCombination data',
    'seperator1': '',
    'seperator2': '',
    'seperator3': '',
}

df = df.reindex(column_order, axis=1)
corr = df.corr()

xcols = column_order[:4]
ycols = column_order[5:-1]
translated_xcols = [translation_dict[col] for col in xcols]
translated_ycols = [translation_dict[col] for col in ycols]

corr = corr.loc[xcols, ycols]


out_path = '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/correlations'

# Plot and save the correlation heatmap
plt.figure(figsize=(18, 9))
ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=translated_ycols, yticklabels=translated_xcols, vmin=-1, vmax=1)
plt.xticks(rotation=45)
major_ticks = np.arange(9) + 0.5
major_ticks = np.delete(major_ticks, 4)
ax.xaxis.set_major_locator(FixedLocator(major_ticks))
ax.minorticks_off()

for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")   # so they "anchor" on the right
    # label.set_x(label.get_position()[0] - 0.25)  # shift a bit left

cbar = ax.collections[0].colorbar
cbar.set_label(r"Correlation of uncalibrated $\eta_{\text{eff}}$", )  # rotated and padded


plt.tight_layout()
plt.savefig(join(out_path, "ref_mc_data_correlation_heatmap.png"))
plt.close()

# Save correlations to a file for later use
corr.to_csv(join(out_path, "ref_mc_data_correlations.csv"))


#Correlation of DA models
paths = {
    "0"   : "/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_DA_0",
    "0.1" : "/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_DA_0.1",
    "0.5" : "/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_DA_0.5",
    "1"   : "/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_DA_1",
    }

taggers = ['OSKaon', ]

df = load_files(paths, taggers)


column_order = ['0_OSKaon_Eta' , '0.1_OSKaon_Eta' , '0.5_OSKaon_Eta' , '1_OSKaon_Eta' ,]

translation_dict = {
    '0_OSKaon_Eta': r'$\lambda=0$',
    '0.1_OSKaon_Eta': r'$\lambda=0.1$',
    '0.5_OSKaon_Eta': r'$\lambda=0.5$',
    '1_OSKaon_Eta': r'$\lambda=1$',
}
df = df.reindex(column_order, axis=1)
corr = df.corr()
xcols = column_order
ycols = column_order
translated_xcols = [translation_dict[col] for col in xcols]
translated_ycols = [translation_dict[col] for col in ycols]
corr = corr.loc[xcols, ycols]
plt.figure(figsize=(8, 6))

ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=translated_ycols, yticklabels=translated_xcols, vmin=-1, vmax=1)
plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

cbar = ax.collections[0].colorbar
cbar.set_label(r"Correlation of uncalibrated $\eta_{\text{eff}}$", )  # rotated and padded

plt.tight_layout()
plt.savefig(join(out_path, "DA_correlation_heatmap.png"))
plt.close()

# Save correlations to a file for later use
corr.to_csv(join(out_path, "DA_correlations.csv"))
