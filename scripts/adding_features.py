import pandas as pd
import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from saver import Saver
import configParameters as config

repoPath = "/ceph/users/molocco/classical-taggers/"
# Decay and tagger type are given as inputs by the user 
eventType = sys.argv[1]

# Get the raw root file 
# Charge and absID are needed for the label 
# List of daughters of the signal B
path_to_tuple = "../Data/merged_mc_bu2jpsik_tuple.root:Tuple/DecayTree;1"

if eventType == "Bs2DsPi": 
    abs_id = 531
    daughters = ["Ds_P","pi_P","Kp_P","Km_P"]

elif eventType == "Bd2JpsiKst":
    abs_id = 511
    daughters = ["mu2","mu1","Kst_P","Jpsi_P"]

elif eventType == "Bu2JpsiK":
    B = "B"
    abs_id = 521
    #path_to_tuple = f"/ceph/users/jroensch/masterthesis/{eventType}/SM_Tuple/{eventType}_nPVsnTracks.root:Tuple/DecayTree;1" # needs to be reproduced in the Repo
    daughters = ["K_P","mu2_P","mu1_P","Jpsi_P"]
    #daughters = ["K_P","Muminus_P","Muplus_P","Jpsi1S_P"]

stepsize = 100000    #the Root file will gel load in chunks

def DeltaQ(df,Mass):
    E =np.sqrt( Mass**2 + df[f"Bp_Tr_T_PX"]**2 + df[f"Bp_Tr_T_PY"]**2 + df[f"Bp_Tr_T_PZ"]**2)
    DeltaQ = np.sqrt( (E + df[f"Bp_ENERGY"])**2  - ((df[f"Bp_Tr_T_PX"] + df[f"Bp_PX"])**2 + (df[f"Bp_Tr_T_PY"] + df[f"Bp_PY"])**2 + (df[f"Bp_Tr_T_PZ"] + df[f"Bp_PZ"])**2 )   ) -df.Bp_M  - Mass
    return(DeltaQ)

start_time = time.time()
run_time = time.time()

#loading_variables = ["Bp_nPVs","Bp_nTracks",f"Bp_TRUEID", f"Bp_Charge", f"Bp_Tr_T_Charge",f"Bp_Tr_T_P",f"Bp_PHI",f"Bp_Tr_T_Phi",f"Bp_BPVZ",f"Bp_BPVY",f"Bp_BPVX",f"Bp_Tr_T_BPVZ",
loading_variables = ["Bp_nPVs","Bp_nTracks",f"Bp_TRUEID", f"Bp_Tr_T_Charge",f"Bp_Tr_T_P",f"Bp_PHI",f"Bp_Tr_T_Phi",f"Bp_BPVZ",f"Bp_BPVY",f"Bp_BPVX",f"Bp_Tr_T_BPVZ", \
                f"Bp_Tr_T_BPVY",f"Bp_Tr_T_BPVX",f"Bp_ETA",f"Bp_Tr_T_Eta",f"Bp_Tr_T_BPVIP",f"Bp_Tr_T_PX",f"Bp_Tr_T_PY",f"Bp_Tr_T_PZ","Bp_P", \
                f"Bp_ENERGY",f"Bp_PX",f"Bp_PY",f"Bp_PZ",f"Bp_Tr_T_TRUEID", "Bp_PT", "Bp_Tr_T_ENERGY", "Bp_Tr_T_PT"  ,"Bp_M", "Bp_Tr_T_ISMUON", \
                "Bp_END_VX","Bp_END_VY","Bp_END_VZ","Bp_Tr_T_X","Bp_Tr_T_Y","Bp_Tr_T_Z", "Bp_Tr_T_M","Bp_Tr_T_PROBNN_E", "Bp_Tr_T_PIDK", "Bp_Tr_T_PIDe", \
                "Bp_Tr_T_PIDmu", "Bp_Tr_T_PIDP",f"Bp_Tr_T_CHI2DOF",f"Bp_Tr_T_PROBNN_P",f"Bp_Tr_T_PROBNN_K",f"Bp_Tr_T_PROBNN_PI",f"Bp_Tr_T_GHOSTPROB",\
                f"Bp_Tr_T_BPVIPCHI2"]+daughters



print('Started reading')
for df in uproot.iterate(path_to_tuple, loading_variables, step_size=stepsize, library = "pd"):
    
    print("-------------------Start Block-----------------------------")
    df.drop(df[abs(df[f"Bp_TRUEID"]) != abs_id ].index , inplace = True) # drop the B mesons or other particles that are not of interest
    for daughter in daughters: #remove daughter
        df.drop(df.loc[df[f"Bp_Tr_T_P"] == df[daughter]].index, inplace = True)
    df.reset_index(inplace=True, drop = False) 
    #Add some needed features
    df.eval(f"Bp_Tr_T_cos_diff_Phi=cos(Bp_PHI-Bp_Tr_T_Phi)", inplace=True)
    df.eval(f"Bp_Tr_T_diff_z = abs(Bp_BPVZ - Bp_Tr_T_BPVZ)" , inplace = True)
    df.eval(f"Bp_Tr_T_PhiDistance =abs(Bp_PHI - Bp_Tr_T_Phi)" , inplace = True)  

    df.eval("Bp_Tr_T_DeltaR= (Bp_ETA - Bp_Tr_T_Eta)**2 + Bp_Tr_T_PhiDistance**2", inplace = True)
    df.eval("diff_P = abs(Bp_P - Bp_Tr_T_P)", inplace = True)
    df.eval("P_proj = Bp_ENERGY*Bp_Tr_T_ENERGY - (Bp_Tr_T_PX*Bp_PX + Bp_Tr_T_PY*Bp_PY +Bp_Tr_T_PZ*Bp_PZ ) ", inplace = True)
    df.eval("t = (Bp_END_VX**2 + Bp_END_VY**2 + Bp_END_VZ**2 - Bp_END_VX*Bp_Tr_T_X - Bp_END_VY*Bp_Tr_T_Y - Bp_END_VZ*Bp_Tr_T_Z) / (Bp_END_VX * Bp_Tr_T_PX + Bp_END_VY * Bp_Tr_T_PY + Bp_END_VZ * Bp_Tr_T_PZ)" , inplace = True)
    df.eval("EVIP = sqrt((Bp_Tr_T_X**2 + Bp_Tr_T_Y**2 + Bp_Tr_T_Z**2) + t**2 * (Bp_Tr_T_PX**2 + Bp_Tr_T_PY**2 + Bp_Tr_T_PZ**2) + 2*t*(Bp_Tr_T_X * Bp_Tr_T_PX + Bp_Tr_T_Y * Bp_Tr_T_PY + Bp_Tr_T_Z * Bp_Tr_T_PZ))", inplace = True)
    df.eval("Bp_Tr_T_absIP = abs(Bp_Tr_T_BPVIP)", inplace = True)
    
    df.eval("Bp_Tr_T_EtaDistance = abs(Bp_ETA - Bp_Tr_T_Eta)", inplace = True)
    df[f"Bp_Tr_T_DeltaQ_Pion"] = DeltaQ(df,139.5706)
    df[f"Bp_Tr_T_DeltaQ_Mu"] = DeltaQ(df,105.65837)
    df[f"Bp_Tr_T_DeltaQ_Electron"] = DeltaQ(df,0.51100)
    df[f"Bp_Tr_T_DeltaQ_Proton"] = DeltaQ(df,938.27208)
    df[f"Bp_Tr_T_DeltaQ_Kaon"] = DeltaQ(df,493.677)
    
    df.eval("Bp_Tr_T_Signal_TagPart_PT = sqrt((Bp_PX + Bp_Tr_T_PX) **2 + (Bp_PY + Bp_Tr_T_PY)**2)", inplace = True)
    df.eval("Bp_Tr_T_eoverP = Bp_Tr_T_Charge/Bp_Tr_T_P", inplace = True)
    df.eval("Bp_Tr_T_absID =abs(Bp_Tr_T_TRUEID)", inplace = True)

    df.eval("EVIP = log(EVIP)", inplace = True)
    df.eval("Bp_Tr_T_IPSig = sqrt(Bp_Tr_T_BPVIPCHI2)" , inplace = True)
    df.eval("P_proj = log(P_proj)", inplace = True)
    
    print(f"Block finished in {round(time.time() - run_time,2)}s")
    print("--------------------End Block------------------------------")
    print()

    run_time = time.time()
    
path = f"{repoPath}root/{eventType}/all_{config.sample_type}_notSelected.root"
with uproot.recreate(path) as f:
    f["DecayTree"] = df

print(f"Modified NTuple saved at {path}")
