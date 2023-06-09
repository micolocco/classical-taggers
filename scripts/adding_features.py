import pandas as pd
import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
import time

repoPath = "/ceph/users/molocco/classical-taggers/"


eventType = sys.argv[1] #For running same script for different eventType 
tagger = sys.argv[2]
training = True

# Decay and tagger type are given as inputs by the user 
eventType = sys.argv[1]
tagger = sys.argv[2]

if len(sys.argv)>3: # For the grid search 
    grid_n = sys.argv[3]
    if len(sys.argv)>4:
        KaonCombiner = True
else: 
    optimized = True

def read_cut(repoPath, tagger, grid_n = None, KaonCombiner = False, optimized = False):
    
    prePath = f"{repoPath}cuts/{tagger}/"
    if grid_n != None: # Load the cut strings 
        if KaonCombiner:
            name = f"recursive_combiner_{grid_n}"
        else:
            name = f"recursive_{grid_n}"
    else:
        if optimized:
            name = f"recursive_optimized"
        else:
            name = f"recursive"
    cut = np.genfromtxt(f"{prePath}{name}.txt", dtype = str, delimiter=",")
    return cut

# Get the raw root file 
# Charge and absID are needed for the label 
# List of daughters of the signal B
path_to_tuple = f"/ceph/users/jroensch/masterthesis/{eventType}/SM_Tuple/{eventType}.root:Tuple/DecayTree;1"

if eventType == "Bs2DsPi": 
    abs_id = 531
    charge = False
    daughters = ["Ds_P","pi_P","Kp_P","Km_P"]

elif eventType == "Bd2JpsiKst":
    abs_id = 511
    charge = False
    daughters = ["Muminus_P","Muplus_P","Kst_P","Jpsi1S_P"]

elif eventType == "Bu2JpsiK":
    B = "B"
    abs_id = 521
    charge = True
    path_to_tuple = f"/ceph/users/jroensch/masterthesis/{eventType}/SM_Tuple/{eventType}_nPVsnTracks.root:Tuple/DecayTree;1" # needs to be reproduced in the Repo
    daughters = ["K_P","Muminus_P","Muplus_P","Jpsi1S_P"]

stepsize = 100000    #the Root file will gel load in chunks

def DeltaQ(df,Mass):
    E =np.sqrt( Mass**2 + df[f"B_Tr_T_PX"]**2 + df[f"B_Tr_T_PY"]**2 + df[f"B_Tr_T_PZ"]**2)
    DeltaQ = np.sqrt( (E + df[f"B_ENERGY"])**2  - ((df[f"B_Tr_T_PX"] + df[f"B_PX"])**2 + (df[f"B_Tr_T_PY"] + df[f"B_PY"])**2 + (df[f"B_Tr_T_PZ"] + df[f"B_PZ"])**2 )   ) -df.B_M  - Mass
    return(DeltaQ)

start_time = time.time()
run_time = time.time()



needed_feature= ["B_nPVs","B_nTracks",f"B_TRUEID", f"B_Charge", f"B_Tr_T_Charge",f"B_Tr_T_P",f"B_Phi",f"B_Tr_T_Phi",f"B_BPVZ",f"B_BPVY",f"B_BPVX",f"B_Tr_T_BPVZ",
                f"B_Tr_T_BPVY",f"B_Tr_T_BPVX",f"B_Eta",f"B_Tr_T_Eta",f"B_Tr_T_BPVIP",f"B_Tr_T_PX",f"B_Tr_T_PY",f"B_Tr_T_PZ","B_P",
                f"B_ENERGY",f"B_PX",f"B_PY",f"B_PZ",f"B_Tr_T_TRUEID", "B_PT",
                 "B_Tr_T_ENERGY", "B_Tr_T_PT"  ,"B_M", "B_Tr_T_ISMUON","B_ENDVX","B_ENDVY","B_ENDVZ","B_Tr_T_X","B_Tr_T_Y","B_Tr_T_Z", "B_Tr_T_M","B_Tr_T_PROBNN_E", "B_Tr_T_PIDK", "B_Tr_T_PIDe", "B_Tr_T_PIDmu", "B_Tr_T_PIDP",f"B_Tr_T_CHI2DOF",f"B_Tr_T_PROBNN_P",f"B_Tr_T_PROBNN_K",f"B_Tr_T_PROBNN_PI",f"B_Tr_T_GHOSTPROB",f"B_Tr_T_BPVIPCHI2"]+daughters



n = True #is needed for saving the file 
print('Starting the reading process')
for df in uproot.iterate(path_to_tuple,needed_feature,step_size=stepsize, library = "pd"):
    
    print("-------------------Start Block-----------------------------")
    df.drop(df[abs(df[f"B_TRUEID"]) != abs_id ].index , inplace = True) # just need the wanted B mesons
    if charge: #getting the labels 
        df["label"] = df[f"B_Charge"] * df[f"B_Tr_T_Charge"]   
    else:
        df["label"] = df[f"B_TRUEID"]/abs(df[f"B_TRUEID"]) * df[f"B_Tr_T_Charge"]  
    
    df.loc[df.label == -1, "label"] = 0 #shiftig the label to 0,1 
    for daughter in daughters: #remove daughter
        df.drop(df.loc[df[f"B_Tr_T_P"] == df[daughter]].index, inplace = True)

    df.reset_index(inplace=True, drop = False) 

    #Add some needed features
    df.eval(f"B_Tr_T_cos_diff_Phi=cos(B_Phi-B_Tr_T_Phi)", inplace=True)
    df.eval(f"B_Tr_T_diff_z = abs(B_BPVZ - B_Tr_T_BPVZ)" , inplace = True)
    df.eval(f"B_Tr_T_PhiDistance =abs(B_Phi - B_Tr_T_Phi)" , inplace = True)  

    df.eval("B_Tr_T_DeltaR= (B_Eta - B_Tr_T_Eta)**2 + B_Tr_T_PhiDistance**2", inplace = True)
    df.eval("diff_P = abs(B_P - B_Tr_T_P)", inplace = True)
    df.eval("P_proj = B_ENERGY*B_Tr_T_ENERGY - (B_Tr_T_PX*B_PX + B_Tr_T_PY*B_PY +B_Tr_T_PZ*B_PZ ) ", inplace = True)
    df.eval("t = (B_ENDVX**2 + B_ENDVY**2 + B_ENDVZ**2 - B_ENDVX*B_Tr_T_X - B_ENDVY*B_Tr_T_Y - B_ENDVZ*B_Tr_T_Z) / (B_ENDVX * B_Tr_T_PX + B_ENDVY * B_Tr_T_PY + B_ENDVZ * B_Tr_T_PZ)" , inplace = True)
    df.eval("EVIP = sqrt((B_Tr_T_X**2 + B_Tr_T_Y**2 + B_Tr_T_Z**2) + t**2 * (B_Tr_T_PX**2 + B_Tr_T_PY**2 + B_Tr_T_PZ**2) + 2*t*(B_Tr_T_X * B_Tr_T_PX + B_Tr_T_Y * B_Tr_T_PY + B_Tr_T_Z * B_Tr_T_PZ))", inplace = True)
    df.eval("B_Tr_T_absIP = abs(B_Tr_T_BPVIP)", inplace = True)
    
    df.eval("B_Tr_T_EtaDistance = abs(B_Eta - B_Tr_T_Eta)", inplace = True)
    df[f"B_Tr_T_DeltaQ_Pion"] = DeltaQ(df,139.5706)
    df[f"B_Tr_T_DeltaQ_Mu"] = DeltaQ(df,105.65837)
    df[f"B_Tr_T_DeltaQ_Electron"] = DeltaQ(df,0.51100)
    df[f"B_Tr_T_DeltaQ_Proton"] = DeltaQ(df,938.27208)
    df[f"B_Tr_T_DeltaQ_Kaon"] = DeltaQ(df,493.677)
    
    df.eval("B_Tr_T_Signal_TagPart_PT = sqrt( (B_PX + B_Tr_T_PX) **2 + (B_PY + B_Tr_T_PY)**2)", inplace = True)
    df.eval("B_Tr_T_eoverP = B_Tr_T_Charge/B_Tr_T_P", inplace = True)
    df.eval("B_Tr_T_absID =abs(B_Tr_T_TRUEID)", inplace = True)

    df.eval("EVIP = log(EVIP)", inplace = True)
    df.eval("B_Tr_T_IPSig = sqrt(B_Tr_T_BPVIPCHI2)" , inplace = True)
    df.eval("P_proj = log(P_proj)", inplace = True)
    cut = read_cut(repoPath, tagger, grid_n = None, KaonCombiner = False, optimized = False)
    df.eval(f"selected_track = {cut}", inplace = True)
    df.selected_track = df.selected_track.astype(int, copy = False) 

    # In case of training = True save only selected tracks. In case of calibration, all tracks are needed for the efficiency computation
    if training:
        df = df.loc[df.selected_track == 1]

    if n: 
        df_save = df
        n = False
    else:
        df_save = pd.concat([df_save,df], ignore_index = True, copy = False)
    print(f"Block finished in {round(time.time() - run_time,2)}s")
    print("--------------------End Block------------------------------")
    print()
    print()

    
    run_time = time.time()
    


with uproot.recreate(f"{repoPath}root/{eventType}withSelected.root") as f:
    f["DecayTree"] = df_save



