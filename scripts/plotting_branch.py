import uproot
from matplotlib import pyplot as plt
import numpy as np
from IPython import embed


path = '/eos/lhcb/user/m/miolocco/FT_NTuple/withUT_MC_2024/2_added_features/Bu2JpsiK/notSelected.root:DecayTree'
#path = '/ceph/users/molocco/classical-taggers/Data/withUT_MC_2024/1_raw/Bu2JpsiK/merged.root:Tuple/DecayTree'
#path = '/ceph-kernel/FlavourTagging/NTuples/OSOptimisationSummer2017/DTT_2016_Reco16Strip28_20170620_kheinicke/DTT_2016_Reco16Strip28_20170625_kheinicke_selected.root:DecayTree;798'
#loading_variables = ['B_Tr_T_PROBNN_MU', 'B_Tr_T_PROBNN_E', 'B_Tr_T_PROBNN_K', 'B_Tr_T_PROBNN_PI', 'B_Tr_T_PROBNN_GHOST', 'B_Tr_T_PIDK', 'B_Tr_T_PIDe', 'B_Tr_T_PIDmu', 'B_Tr_T_PIDP', 'B_Tr_T_TRUEID']#['B_OSMuonDev_TagPartsFeature_IPsig']#['B_Tr_T_BVIPSig']
loading_variables = ['B_Tr_T_PIDK', 'B_Tr_T_PIDP', 'B_Tr_T_TRUEID']#['B_OSMuonDev_TagPartsFeature_IPsig']#['B_Tr_T_BVIPSig']

df = uproot.open(path).arrays(loading_variables,library = "pd" ) # load all data
#plt.hist(df['B_Tr_T_BVIPSig'], bins=100, color='r', alpha=0.5)
#plt.hist(df['B_Tr_T_IPErr'], bins=100, color='r', alpha=0.5)
#plt.hist(df.B_Tr_T_IPErr[df['B_Tr_T_IPErr']<10], bins=100, color='r', alpha=0.5)
#var = 'B_Tr_T_BPVIPCHI2'
#plt.hist(df[var][(df['B_Tr_T_IPErr']<10) & (df['B_Tr_T_BPVIP']<3)], bins=100, color='r', alpha=0.5)
#plt.hist(df[var][(df['B_Tr_T_BPVIPCHI2']<10)], bins=100, color='r', alpha=0.5)
#plt.title(var)
#plt.show()
#embed()


#plt.figure(figsize=(35,35))
'''

for i, col in enumerate(df.columns.to_list()):
    plt.subplot(3, 3, i + 1)
    plt.hist(df[col][df['B_Tr_T_TRUEID'].abs()==211], density = True, bins=100, label = "pion", color='r', alpha=0.5)
    plt.hist(df[col][df['B_Tr_T_TRUEID'].abs()==321], density = True,bins=100, label = "kaon", color='b', alpha=0.5)
    plt.hist(df[col][df['B_Tr_T_TRUEID'].abs()==2212], density = True,bins=100, label = "proton", color='g', alpha=0.5)

   # plt.hist(df[col][df['label']==1][df['selected']==1], density = True, bins=100, label = "post select, label = 1", color='r', alpha=0.2)
   # plt.hist(df[col][df['label']==0], density = True, bins=100, label = "label = 0",color='b', alpha=0.5)
   # plt.hist(df[col][df['label']==1], density = True, bins=100, label = "label = 1",color='b', alpha=0.2)
    #plt.hist(df[col], bins=100, color='r', alpha=0.5)
    plt.legend()
    plt.title(col)
    plt.tight_layout()
#plt.show()
plt.savefig(f"PIDs_byTRUEID.pdf")'''

plt.scatter(df['B_Tr_T_PIDK'][df['B_Tr_T_TRUEID'].abs()==211], df['B_Tr_T_PIDP'][df['B_Tr_T_TRUEID'].abs()==211], label = "pion", color='r', alpha=0.5)
plt.scatter(df['B_Tr_T_PIDK'][df['B_Tr_T_TRUEID'].abs()==321], df['B_Tr_T_PIDP'][df['B_Tr_T_TRUEID'].abs()==321], label = "kaon", color='b', alpha=0.5)
plt.scatter(df['B_Tr_T_PIDK'][df['B_Tr_T_TRUEID'].abs()==2212], df['B_Tr_T_PIDP'][df['B_Tr_T_TRUEID'].abs()==2212], label = "proton", color='g', alpha=0.5)
plt.legend()
plt.xlabel('B_Tr_T_PIDK')
plt.ylabel('B_Tr_T_PIDP')


#plt.savefig(f"PIDs_scatter.pdf")
plt.show()