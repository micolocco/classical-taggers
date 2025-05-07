import argparse
from pprint import pprint
import scripts.pyTorchTraining as pyTrain
from scripts.train_tagger import training_pipeline
from scripts.train_tagger import read_files
from scripts.NNModel import NeuralNetwork
import datetime
import multiprocessing as mp
import sys
import threading

def train_with_log(train_df, val_df, vars, weight_label, BID, val_files, pre_path, treename, tagger, seed, features, 
                    config, decay_type, clean, repo, data_type, weight_type, num_threads):
    target_path = pre_path + f'/{seed}/{config}/{weight_type}/training/'

    logfile = f'{target_path}/training_log.log'

    with open(logfile, 'w') as f:  
        sys.stdout = f
        try:
            training_pipeline(train_df=train_df, val_df=val_df, vars=vars, weight_label=weight_label, BID=BID, 
                target_path=target_path, treename=treename, tagger=tagger, seed=seed, features=features, 
                config=config, decay_type=decay_type, clean=clean, repo=repo, data_type=data_type, 
                weight_type=weight_type, num_threads=num_threads)
        finally:
            sys.stdout = sys.__stdout__  # Restore original stdout
    
#Define a custom stdout class to handle thread-local output to different log files
class ThreadLocalStdout:
    def __init__(self):
        self.local = threading.local()

    def set_log_file(self, filename):
        self.local.log_file = open(filename, 'w')

    def write(self, message):
        if hasattr(self.local, 'log_file'):
            self.local.log_file.write(message)
            self.local.log_file.flush()
        else:
            sys.__stdout__.write(message)

    def flush(self):
        if hasattr(self.local, 'log_file'):
            self.local.log_file.flush()
        else:
            sys.__stdout__.flush()

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a batch of taggers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training_data', help='Files of training data', nargs='+')
    parser.add_argument('--validation_data', help='Files of validation data', nargs='+')
    parser.add_argument('--pre_path', help='Name of the output dir', type=str, ) #/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--seed', help='Random seed', default=45, type = int) 
    parser.add_argument('--features', help='Input features for NN training', default='union') 
    parser.add_argument('--config', help='Config yaml', type=str, nargs='+') 
    parser.add_argument('--decay_type', help='Event decay', type=str)
    parser.add_argument('--clean', help='Decide whatever cleaning the directories before running, w=False, a=True', action='store_true')
    parser.add_argument('--repo', help="Path to repository")
    parser.add_argument('--data_type', help="Type of Data used, MC or Data",choices=('MC', 'Data'))
    parser.add_argument('--weight_type', help="Type of sample weight to be used for training on data", choices=('signal_weights', 'pdf_ratio', 'ones'))
    parser.add_argument('--training_logs', help='Path to the log files of each individual training', type=str, nargs='+')
    

    cfg = parser.parse_args()
    sys.stdout = ThreadLocalStdout()

    pprint(cfg)

    BID = 'B_ID' if cfg.data_type == 'Data' else 'B_TRUEID'

    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)


    if cfg.data_type == 'Data':
        print(f'features are {features}')
        for i in range(len(features)):
            features[i] = features[i].replace("BPVIP", "OWNPVIP")
            features[i] = features[i].replace("B_TRUEID", "B_ID")

    vars = features + [BID,'selected', 'label',f"{cfg.tagger}_TagDec"] #'B_Tr_T_Charge',
    if cfg.data_type == 'Data':
        weight_label = cfg.weight_type
        if weight_label != 'ones':
            vars = vars + [weight_label]



    #Reading Data from files
    print(f'Reading of training files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.train_files)} files.", flush=True)
    train_df = read_files(cfg.train_files, vars = vars, treename=cfg.treename)
    print(f'Reading of training files ends {datetime.datetime.now().strftime("%H:%M:%S")}')

    print(f'Reading of validation files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.val_files)} files.", flush=True)
    val_df = read_files(cfg.val_files, vars = vars, treename=cfg.treename)
    print(f'Reading of validation files ends {datetime.datetime.now().strftime("%H:%M:%S")}')

    mp.set_start_method('spawn')
    threads = []
    for config in cfg.config:
        print(f'Config file used: {config}: Training begins {datetime.datetime.now().strftime("%H:%M:%S")}')

        logfile = next((log for log in cfg.training_logs if config in log), None)
        p = mp.Process(target=train_with_log, args=(train_df, val_df, vars, weight_label, BID, cfg.pre_path, cfg.treename, cfg.tagger, cfg.seed, features,
                                                          config, cfg.decay_type, cfg.repo, cfg.data_type, cfg.weight_type, 1, cfg.clean, logfile))
        threads.append(p)
        p.start()

        
    for p in threads:
        p.join()
    

