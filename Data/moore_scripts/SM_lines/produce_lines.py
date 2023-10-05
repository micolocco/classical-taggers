import sys
import os 

i = int(sys.argv[1])
eventType = sys.argv[2]

dir_path = f"/ceph/users/molocco/classical-taggers/Data/moore_scripts/SM_lines/"
if not os.path.exists(f"{dir_path}{eventType}"):
    os.makedirs(f"{dir_path}{eventType}")

for n in range(i):
    fin = open(f"{dir_path}/template.txt" , "rt")
    fout = open(f"{dir_path}{eventType}/line_SnakeMake_{n}.py", "wt")
    for line in fin:
        fout.write(line.replace("@",f"{n}").replace("eventType",f"{eventType}"))

    fout.close()
    fin.close()
