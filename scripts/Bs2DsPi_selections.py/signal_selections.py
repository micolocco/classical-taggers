import numpy as np
import uproot
import argparse
import os

from IPython import embed
import helper


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

    AP_preselections = [ # Apply pre-selections used by b2oc https://gitlab.cern.ch/lhcb-b2oc/analyses/b2dx-early-measurements/-/blob/master/BranchesAndSelection.py#L268
            '(piplus_PIDK<0.)',
            '(piplus_P>2000.0 & piplus_P<150000.0 & piplus_PT>250.0 & piplus_PT<45000.0 & piplus_ETA>2.0 & piplus_ETA<5.0)',
            '(hplus_P>2000.0 & hplus_P<150000.0 & hplus_PT>250.0 & hplus_PT<45000.0 & hplus_ETA>2.0 & hplus_ETA<5.0)',
            '(hminus_P>2000.0 & hminus_P<150000.0 & hminus_PT>250.0 & hminus_PT<45000.0 & hminus_ETA>2.0 & hminus_ETA<5.0)',
            '(piminus_P>2000.0 & piminus_P<150000.0 & piminus_PT>250.0 & piminus_PT<45000.0 & piminus_ETA>2.0 & piminus_ETA<5.0)',
            ],

    AP_preselect_variables = ['piplus_PIDK', 'piplus_P', 'piplus_PT', 'piplus_ETA', 'hplus_P', 'hplus_PT', 'hplus_ETA',
                         'hminus_P', 'hminus_PT', 'hminus_ETA', 'piminus_P', 'piminus_PT', 'piminus_ETA']


   optimize_tree_reading(cfg.raw, AP_preselect_variables)
        
    # Save the selected tracks into NTuples
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    print(df.columns)

    with uproot.recreate(f"{cfg.output}") as file:
        file[cfg.treename] = df
    print(f"Pre-selections applied. NTuple saved at {cfg.output}")

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
            'lab1_MINIPCHI2_Log':'log(lab1_BPVIPCHI2)',
                'lab1_CosTheta':'GetTheS(lab0_MASS,lab0_PX,lab0_PY,lab0_PZ,lab1_MASS,lab1_PX,lab1_PY,lab1_PZ)',
                'lab345_MIN_PT':'(min(min(lab3_PT, lab4_PT), lab5_PT))',
                'lab345_MIN_MINIPCHI2_Log':'log(min(min(lab3_IPCHI2_OWNPV, lab4_IPCHI2_OWNPV), lab5_IPCHI2_OWNPV))',
                'lab1345_TRACK_GhostProb':'max(max(lab1_TRACK_GhostProb, lab3_TRACK_GhostProb), max(lab4_TRACK_GhostProb, lab5_TRACK_GhostProb))',
                }]







'''   
