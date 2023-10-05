import numpy as np 
import sys 
import os

n = int(sys.argv[1])
eventType = sys.argv[2]

dir_path = f"/ceph/users/molocco/classical-taggers/Data/moore_scripts/LFNs/{eventType}/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
LFN_list = open(f"{dir_path}/All_LFN.txt","r").read().split("\n")
    
list_len = len(LFN_list)

LFNs = np.array_split(LFN_list,n)
counter = 0
for i in range(n):
    with open(f"{dir_path}/LFN_{i}.txt","w") as file:
        for lfn in LFNs[i]:
            file.write("%s\n"%lfn)


