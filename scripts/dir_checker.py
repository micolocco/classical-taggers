import numpy as np 
import os 

def check_directories (repoPath,directory_list, eventType, tagger):
    for dir in directory_list:
        dir_path = f"{repoPath}/{dir}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(f"{dir_path}/{eventType}"):
            os.makedirs(f"{dir_path}/{eventType}")
        if not os.path.exists(f"{dir_path}/{eventType}/{tagger}"):
            os.makedirs(f"{dir_path}/{eventType}/{tagger}")
