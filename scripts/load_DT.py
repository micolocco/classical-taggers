import DT_utils
import pickle
from collections import defaultdict
import os
from IPython import embed


output_path = "/ceph/users/molocco/Data/withUT_MC_2024/DT_outputs/notSamePV_noOSP_SSK/balanced/"
print("Loading the model...")
with open(f"{output_path}/decision_tree_model.pkl", "rb") as f:
    clf = pickle.load(f)
print(f"Model loaded successfully! Type of clf: {type(clf)}")

features_added = ['B_Tr_T_minPhiDistance', 'B_Tr_T_cos_PhiDistance', 'B_Tr_T_PhiDistance', 'B_Tr_T_diff_z', 'B_Tr_T_DeltaR', 'diff_P', 'P_proj', 't', 'EVIP', 'B_Tr_T_absOWNPV_IP', 'B_Tr_T_EtaDistance', 'B_Tr_T_DeltaQ_Pion', 'B_Tr_T_DeltaQ_Muon', 'B_Tr_T_DeltaQ_Electron', 'B_Tr_T_DeltaQ_Proton', 'B_Tr_T_DeltaQ_Kaon', 'B_Tr_T_Signal_TagPart_PT', 'B_Tr_T_eoverP', 'B_Tr_T_OWNPVIPSig', 'logEVIP', 'logP_proj', 'B_Tr_T_atanPT_PZ']
features_noMC = [
        'B_Tr_T_TRACKISLONG',
        'B_Tr_T_OWNPVIP',
        'B_Tr_T_OWNPVIPCHI2',
        'B_Tr_T_Charge',
        'B_Tr_T_ISMUON',
        'B_Tr_T_ENERGY',
        'B_Tr_T_Eta',
        'B_Tr_T_MINIP',
        'B_Tr_T_MINIPChi2',
        'B_Tr_T_P',
        'B_Tr_T_PT',
        'B_Tr_T_PIDK',
        'B_Tr_T_PIDe',
        'B_Tr_T_PIDmu',
        'B_Tr_T_PIDP',
        'B_Tr_T_PROBNN_GHOST',
        'B_Tr_T_PROBNN_E',
        'B_Tr_T_PROBNN_K',
        'B_Tr_T_PROBNN_P',
        'B_Tr_T_PROBNN_MU',
        'B_Tr_T_PROBNN_PI',
        'B_Tr_T_firstX',
        'B_Tr_T_firstY',
        'B_Tr_T_firstZ',
        'B_Tr_T_firstTX',
        'B_Tr_T_firstTY',
        'B_Tr_T_OWNPV_X',
        'B_Tr_T_OWNPV_XERR',
        'B_Tr_T_OWNPV_Y',
        'B_Tr_T_OWNPV_YERR',
        'B_Tr_T_OWNPV_Z',
        'B_Tr_T_OWNPV_ZERR',
        'B_Tr_T_Phi',
        'B_Tr_T_M',
        'B_Tr_T_CHI2DOF',
        'B_Tr_T_GHOSTPROB',
        'B_Tr_T_PX',
        'B_Tr_T_PY',
        'B_Tr_T_PZ',
        'B_Tr_T_X',
        'B_Tr_T_Y',
        'B_Tr_T_Z',
        'B_Tr_T_IPChi2BVTX',
        'B_Tr_T_IPBVTX',]
        
features = features_noMC + features_added
# Get all decision paths from the classifier
paths = DT_utils.get_decision_paths(clf, features)

# Group the paths by class label
paths_by_class = defaultdict(list)
for conditions, label in paths:
    # Combine conditions using AND for a single path.
    combined = " & ".join(conditions)
    paths_by_class[label].append(combined)

# Write the conditions for each class into separate files.
os.makedirs(f'{output_path}/cuts', exist_ok=True)
for label, conditions_list in paths_by_class.items():
    # Create a file name based on the class label.
    filename = os.path.join(f'{output_path}/cuts', f"{label}_preselection.txt")
    with open(filename, "w") as f:
        # If multiple paths lead to the same class, separate them with OR.
        f.write("\nOR\n".join(conditions_list))
    print(f"Saved cuts for class '{label}' in {filename}")
