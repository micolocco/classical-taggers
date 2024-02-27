import os 

def data_make_dir(sample_type):
    decays = ["Bd2JpsiKst" , "Bs2DsPi", "Bu2JpsiK", "Bd2DmPi"]
    folders = ['1_raw', '2_added_features', '3_selected']
    path = '/eos/lhcb/user/m/miolocco/FT_NTuple'
    #if not os.path.exists(f"Data"):
    #    os.makedirs(f"Data")
    if not os.path.exists(f"{path}/{sample_type}"):
        os.makedirs(f"{path}/{sample_type}")
    for folder in folders:
        if not os.path.exists(f"{path}/{sample_type}/{folder}"):
            os.makedirs(f"{path}/{sample_type}/{folder}")
        for decay in decays:
            if not os.path.exists(f"{path}/{sample_type}/{folder}/{decay}"):
                os.makedirs(f"{path}/{sample_type}/{folder}/{decay}")


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