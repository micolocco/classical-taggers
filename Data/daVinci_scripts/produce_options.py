import sys
import os
i = int(sys.argv[1])
eventType = sys.argv[2]

dir_path = f"/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/{eventType}/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(f"{dir_path}options_SM/"):
    os.makedirs(f"{dir_path}options_SM/")
if not os.path.exists(f"{dir_path}output/"):
    os.makedirs(f"{dir_path}output/")
if not os.path.exists(f"{dir_path}logs/"):
    os.makedirs(f"{dir_path}logs/")
if not os.path.exists(f"{dir_path}RootRaw/"):
    os.makedirs(f"{dir_path}RootRaw/")



for n in range(i):
    fin = open(f"/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/template.txt" , "rt")
    fout = open(f"{dir_path}options_SM/options_{n}.yaml", "wt")
    for line in fin:
        fout.write(line.replace("@",f"{n}").replace("eventType", f"{eventType}"))
    fout.close()
    fin.close()
