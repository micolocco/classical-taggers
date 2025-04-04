import uproot 
import pandas as pd
from IPython import embed

loading_variables = ["B_BKGCAT", "B_Tr_T_absID", "B_Tr_T_Origin_Flag"]
f = "/ceph/users/molocco/Data/withUT_MC_2024/2_added_features/Bd2JpsiKst/00267659_00000001_1.mc.root"
with uproot.open("{}".format(f)) as _f:
    df = _f["Tuple/DecayTree"].arrays(loading_variables, library="pd")
embed()
df = df[(df.B_Tr_T_absID == 211) & (df.B_Tr_T_Origin_Flag == 1)] #SSPiom
