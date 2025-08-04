import ROOT
import os
from IPython import embed
import re
import argparse


def apply_label_map(strings, label_map):
    """Replace labX_ with particle name using label_map in list of strings."""
    result = []
    for s in strings:
        for lab, name in label_map.items():
            s = re.sub(rf'\b{lab}_', f'{name}_', s)
        result.append(s)
    return result

def apply_preselections_and_save_all_branches(infile, tree_name, outfile, preselections, preselect_variables):
    # Open input ROOT file
    inFile = ROOT.TFile.Open(infile)
    inTree = inFile.Get(tree_name)

    # Set all branches to inactive to save memory
    inTree.SetBranchStatus("*", 0)

    # Activate only branches needed for selection
    for var in preselect_variables:
        inTree.SetBranchStatus(var, 1)

    # Create output file and clone tree structure (empty)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    outFile = ROOT.TFile(outfile, "RECREATE")

    # Reactivate all branches for full event copy (but reading one-by-one)
    inTree.SetBranchStatus("*", 1)
    outTree = inTree.CloneTree(0)

    # Compile the cut expression into a TTreeFormula
    cut_expr = " && ".join(preselections)
    formula = ROOT.TTreeFormula("cut", cut_expr, inTree)

    # Loop over entries
    nEntries = inTree.GetEntries()
    for i in range(nEntries):
        inTree.GetEntry(i)

        # Evaluate preselections
        formula.GetNdata()  # Required by ROOT to update internal state
        if formula.EvalInstance():
            outTree.Fill()

        if i % 10000 == 0:
            print(f"{i}/{nEntries} processed...")

    # Write and close
    outFile.cd()
    outTree.Write()

    print(f"Saved {outTree.GetEntries()} selected entries to {outfile}")
    outFile.Close()
    inFile.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--raw', help='Raw file where cuts need to be applied', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='BdToDsmPi_DsmToKpKmPim/DecayTree')


    cfg = parser.parse_args()

    from pprint import pprint
    pprint(cfg)

    # Define label map to adapt to different naming convention DsPi analysis production VS phis Analysis production
    label_map = {
        'lab0': 'B',
        'lab1': 'piplus',
        'lab2': 'Ds',
        'lab3': 'hplus',
        'lab4': 'hminus',
        'lab5': 'piminus',
    }


    AP_preselections_b2oc = [
    '(lab1_PID_K < 0.)',
    '(lab1_P > 2000.0 & lab1_P < 150000.0 & lab1_PT > 250.0 & lab1_PT < 45000.0 & lab1_ETA > 2.0 & lab1_ETA < 5.0)',
    '(lab3_P > 2000.0 & lab3_P < 150000.0 & lab3_PT > 250.0 & lab3_PT < 45000.0 & lab3_ETA > 2.0 & lab3_ETA < 5.0)',
    '(lab4_P > 2000.0 & lab4_P < 150000.0 & lab4_PT > 250.0 & lab4_PT < 45000.0 & lab4_ETA > 2.0 & lab4_ETA < 5.0)',
    '(lab5_P > 2000.0 & lab5_P < 150000.0 & lab5_PT > 250.0 & lab5_PT < 45000.0 & lab5_ETA > 2.0 & lab5_ETA < 5.0)',
]


    AP_preselect_variables_b2oc = [
    'lab1_PID_K', 'lab1_P', 'lab1_PT', 'lab1_ETA',
    'lab3_P', 'lab3_PT', 'lab3_ETA',
    'lab4_P', 'lab4_PT', 'lab4_ETA',
    'lab5_P', 'lab5_PT', 'lab5_ETA',
]
    AP_preselections = apply_label_map(AP_preselections_b2oc, label_map)
    AP_preselect_variables = apply_label_map(AP_preselect_variables_b2oc, label_map)
    print(f"Preselections: {AP_preselections}")
    print(f"Variables used for applying preselections: {AP_preselect_variables}")
    # Save the selected tracks into NTuples
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)

    apply_preselections_and_save_all_branches(
        infile=cfg.raw,
        tree_name=cfg.treename,
        outfile=cfg.output,
        preselections=AP_preselections,
        preselect_variables=AP_preselect_variables
    )

'''
     # Collect features
            event_data = [
                inTree.lab0_DIRA_OWNPV_Transformed,
                inTree.lab0_MINIPCHI2_Log,
                inTree.lab0_RFD,
                inTree.lab0_VCHI2NDOF_Log,
                inTree.lab0_LifetimeFit_VCHI2NDOF_Log,
                inTree.lab2_DIRA_ORIVX_Transformed,x
                inTree.lab2_MINIPCHI2_Log,
                inTree.lab2_RFD,
                inTree.lab2_VCHI2NDOF_Log,
                inTree.lab1_MINIPCHI2_Log,
                inTree.lab1_PT,
                inTree.lab1_CosTheta,
                inTree.lab345_MIN_PT,
                inTree.lab345_MIN_MINIPCHI2_Log,
                inTree.lab1345_TRACK_GhostProb
            ]
    'variables_to_define': [{
           'lab0_DIRA_OWNPV_Transformed': '((lab0_DIRA_OWNPV > 0.0 && lab0_DIRA_OWNPV <= 1.0) ? -log(1.0 + 1.0e-6 - lab0_DIRA_OWNPV) : (lab0_DIRA_OWNPV <= 0.0 && lab0_DIRA_OWNPV >= -1.0) ? log(1.0 + 1.0e-6 + lab0_DIRA_OWNPV) : 0.0)', 
           'lab0_MINIPCHI2_Log':'log(lab0_BPVIPCHI2)',
            'lab0_RFD':'(sqrt(pow(lab0_ENDVERTEX_X - lab0_OWNPV_X,2) + pow(lab0_ENDVERTEX_Y - lab0_OWNPV_Y,2)))',
            'lab0_VCHI2NDOF_Log':'log(lab0_END_VCHI2DOF)',
            'lab0_LifetimeFit_VCHI2NDOF_Log':'log(lab0_DTF_LifetimeFit_CHI2DOF )',
            'lab2_DIRA_ORIVX_Transformed': '((lab2_DIRA_OWNPV > 0.0 && lab2_DIRA_OWNPV <= 1.0) ? -log(1.0 + 1.0e-6 - lab2_DIRA_OWNPV) : (lab2_DIRA_OWNPV <= 0.0 && lab2_DIRA_OWNPV >= -1.0) ? log(1.0 + 1.0e-6 + lab2_DIRA_OWNPV) : 0.0)',
            'lab2_MINIPCHI2_Log':'log(lab2_BPVIPCHI2)',
            'lab2_RFD':'(sqrt(pow(lab2_ENDVERTEX_X - lab2_OWNPV_X,2) + pow(lab2_ENDVERTEX_Y - lab2_OWNPV_Y,2)))',
         'lab2_VCHI2NDOF_Log':'log(lab2_END_VCHI2DOF)',
            'lab1_MINimport ROOT
import os

def apply_preselections_and_save_all_branches(infile, tree_name, outfile, preselections, preselect_variables):
    # Open input ROOT file
    inFile = ROOT.TFile.Open(infile)
    inTree = inFile.Get(tree_name)

    # Set all branches to inactive to save memory
    inTree.SetBranchStatus("*", 0)

    # Activate only branches needed for selection
    for var in preselect_variables:
        inTree.SetBranchStatus(var, 1)

    # Create output file and clone tree structure (empty)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    outFile = ROOT.TFile(outfile, "RECREATE")
    outTree = inTree.CloneTree(0)

    # Reactivate all branches for full event copy (but reading one-by-one)
    inTree.SetBranchStatus("*", 1)

    # Compile the cut expression into a TTreeFormula
    cut_expr = " && ".join(preselections)
    formula = ROOT.TTreeFormula("cut", cut_expr, inTree)

    # Loop over entries
    nEntries = inTree.GetEntries()
    for i in range(nEntries):
        inTree.GetEntry(i)

        # Evaluate preselections
        formIPCHI2_Log':'log(lab1_BPVIPCHI2)',
                'lab1_CosTheta':'GetTheS(lab0_MASS,lab0_PX,lab0_PY,lab0_PZ,lab1_MASS,lab1_PX,lab1_PY,lab1_PZ)',
                'lab345_MIN_PT':'(min(min(lab3_PT, lab4_PT), lab5_PT))',
                'lab345_MIN_MINIPCHI2_Log':'log(min(min(lab3_IPCHI2_OWNPV, lab4_IPCHI2_OWNPV), lab5_IPCHI2_OWNPV))',
                'lab1345_TRACK_GhostProb':'max(max(lab1_TRACK_GhostProb, lab3_TRACK_GhostProb), max(lab4_TRACK_GhostProb, lab5_TRACK_GhostProb))',
                }]







'''   
