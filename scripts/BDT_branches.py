config = {}
from MINIBDTSelection_HelperFunctions import *
import os

config = {
    'variables_to_define': 
    [{
        "lab0_CHI2DOF":"B_OWNPV_CHI2DOF", #['B_ENDV_CHI2DOF','B_OWNPV_CHI2DOF','B_Tr_T_CHI2DOF',
        "lab0_IPCHI2_OWNPV":"B_OWNPV_IPCHI2", 
        "eventNumber":"EVENTNUMBER",
        "lab0_DIRA_OWNPV_Transformed":"((B_OWNPV_DIRA > 0.0 && B_OWNPV_DIRA <= 1.0) ? -log(1.0 + 1.0e-6 - B_OWNPV_DIRA) : (B_OWNPV_DIRA <= 0.0 && B_OWNPV_DIRA >= -1.0) ? log(1.0 + 1.0e-6 + B_OWNPV_DIRA) : 0.0)", 
        "lab0_MINIPCHI2_Log":"log(B_MIN_OWNPV_IPCHI2)",
        "lab0_RFD":"(sqrt(pow(B_ENDV_X - B_OWNPV_X,2) + pow(B_ENDV_Y - B_OWNPV_Y,2)))",
        "lab0_VCHI2NDOF_Log":"log(B_ENDV_CHI2DOF)", 
        "lab0_LifetimeFit_VCHI2NDOF_Log":"log(B_DTF_PV_B_CHI2DOF)",
        "lab2_DIRA_ORIVX_Transformed":"((Ds_OWNPV_DIRA > 0.0 && Ds_OWNPV_DIRA <= 1.0) ? -log(1.0 + 1.0e-6 - Ds_OWNPV_DIRA) : (Ds_OWNPV_DIRA <= 0.0 && Ds_OWNPV_DIRA >= -1.0) ? log(1.0 + 1.0e-6 + Ds_OWNPV_DIRA) : 0.0)", 
        "lab2_MINIPCHI2_Log":"log(Ds_MIN_OWNPV_IPCHI2)",
        "lab2_RFD":"(sqrt(pow(Ds_ENDV_X - Ds_OWNPV_X,2) + pow(Ds_ENDV_Y - Ds_OWNPV_Y,2)))",
        "lab2_VCHI2NDOF_Log":"log(Ds_ENDV_CHI2DOF)", 
        "lab1_MINIPCHI2_Log":"log(piplus_MINIPCHI2)", 
        "lab1_PT":"piplus_PT",
        "lab1_CosTheta":"GetTheS(B_M,B_PX,B_PY,B_PZ,piplus_M,piplus_PX,piplus_PY,piplus_PZ)", 
        "lab345_MIN_PT":"(min(min(hplus_PT, hminus_PT), piminus_PT))", 
        "lab345_MIN_MINIPCHI2_Log":"log(min(min(hplus_OWNPV_IPCHI2, hminus_OWNPV_IPCHI2), piminus_OWNPV_IPCHI2))",
        "lab1345_TRACK_GhostProb": "max(max(piplus_TRGHOSTPROB, hplus_TRGHOSTPROB), max(hminus_TRGHOSTPROB, piminus_TRGHOSTPROB))"
    }],
}