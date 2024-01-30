import os 

def make_dir(repoPath, sample_type, decay):
    '''Function to create/check output directory.
    The path will be of format:
     <yourRepoPath>/savedModels/<sample_type>/<decay>/<tagger>
    example: <yourRepoPath>/savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/
    '''
    decays = ["Bd2JpsiKst" , "Bs2DsPi", "Bu2JpsiK"]
    folder = 'savedModels'
    if not os.path.exists(f"{repoPath}{folder}/{sample_type}"):
        os.makedirs(f"{repoPath}{folder}/{sample_type}")
    for decay in decays:
        if decay == "Bu2JpsiK":
            if not os.path.exists(f"{repoPath}{folder}/{sample_type}/{decay}"):
                os.makedirs(f"{repoPath}{folder}/{sample_type}/{decay}/OSKaon")
                os.makedirs(f"{repoPath}{folder}/{sample_type}/{decay}/OSMuon")
                os.makedirs(f"{repoPath}{folder}/{sample_type}/{decay}/OSElectron")
        if decay == "Bs2DsPi":
            if not os.path.exists(f"{repoPath}{folder}/{sample_type}/{decay}"):
                os.makedirs(f"{repoPath}{folder}/{sample_type}/{decay}/SSKaon")
        if decay == "Bd2JpsiKst":
            if not os.path.exists(f"{repoPath}{folder}/{sample_type}/{decay}"):
                os.makedirs(f"{repoPath}{folder}/{sample_type}/{decay}/SSPion")
                os.makedirs(f"{repoPath}{folder}/{sample_type}/{decay}/SSProton")