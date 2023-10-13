import Functors as F
from DaVinci import Options, make_config
from FunTuple import FunctorCollection, FunTuple_Particles as Funtuple
from PyConf.Algorithms import ParticleTaggerAlg, ParticleContainerMerger
from FunTuple.functorcollections import Kinematics, EventInfo ,MCKinematics, MCHierarchy
from PyConf.reading import get_odin 
from DaVinciMCTools import MCTruthAndBkgCat #changed
from PyConf.reading import get_particles, get_pvs
from RecoConf.event_filters import require_pvs



def main(options: Options):

    fields = {
        "B" : "[B+ -> (J/psi(1S) -> mu+ mu-) K+]CC",
        "Jpsi1S" : "[B+ -> ^(J/psi(1S) -> mu+ mu-) K+]CC",
        "K": "[B+ -> (J/psi(1S) -> mu+ mu-) ^K+]CC",
        "Muplus": "[B+ -> (J/psi(1S) -> ^mu+ mu-) K+]CC",
        "Muminus": "[B+ -> (J/psi(1S) -> mu+ ^mu-) K+]CC",
    }

    pvs = get_pvs()

    variables_all = Kinematics() #Mass,P,PT,P{XYZ},ENERGY
    all_variables_add =  FunctorCollection({
        "BPVIP": F.BPVIP(pvs),
        "MINIP":F.MINIP(Vertices = pvs), 
        "MINIPCHI2":F.MINIPCHI2(Vertices = pvs), 
        "ID": F.PARTICLE_ID,
       # "Charge": F.CHARGE,
        "ISMUON": F.ISMUON,
        "OBJECT_KEY" : F.OBJECT_KEY,
        "Phi" : F.PHI,
        "Eta" : F.ETA,
        "CHI2DOF": F.CHI2DOF,
        "BPVX":F.BPVX(pvs),
        "BPVY":F.BPVY(pvs),
        "BPVZ":F.BPVZ(pvs),
    })

    B2JpsiK = get_particles('/Event/HLT2/Hlt2_Bu2JpsiK_Jpsi2MuMu/Particles')
    longPions = get_particles('/Event/HLT2/Hlt2_Bu2JpsiK_Jpsi2MuMu/LongTaggingParticles/Particles')
    upstreamPions = get_particles('/Event/HLT2/Hlt2_Bu2JpsiK_Jpsi2MuMu/UpstreamTaggingParticles/Particles')

    tagging_container = ParticleContainerMerger(
        InputContainers=[longPions, upstreamPions]).OutputContainer

    tagAlg = ParticleTaggerAlg(
        Input=B2JpsiK, TaggingContainer=tagging_container, OutputLevel=3)
    tagAlg_rels = tagAlg.OutputRelations

    int_class = int(0)
    # helper lambda function
    mctruth = MCTruthAndBkgCat(B2JpsiK)
    mctruth_tag = MCTruthAndBkgCat(tagging_container)
    
    MCTRUTH = lambda func: F.MAP_INPUT(Functor=func, Relations=mctruth.MCAssocTable)
    MCTRUTH_TAG = lambda func: F.MAP_INPUT(Functor=func, Relations=mctruth_tag.MCAssocTable)

    MCMOTHER_ID = lambda n: F.MAP_INPUT_ARRAY(Functor=F.VALUE_OR(0) @ MCTRUTH_TAG(F.MC_MOTHER(n, F.PARTICLE_ID)), Relations=tagAlg_rels)
    MCMOTHER_KEY = lambda n: F.MAP_INPUT_ARRAY(Functor=F.VALUE_OR(0) @ MCTRUTH_TAG(F.MC_MOTHER(n, F.OBJECT_KEY)), Relations=tagAlg_rels)



    variables_Bu = FunctorCollection({
        "nPVs":F.VALUE_OR(0) @ F.SIZE_OF @ F.MAP(F.Z_COORDINATE @ F.POSITION) @ F.TES(pvs),
        "nTracks": F.VALUE_OR(0) @ F.SIZE_OF @ F.MAP_INPUT_ARRAY(Functor= F.PX, Relations = tagAlg_rels),
        "Tr_T_BPVIP" : F.MAP_INPUT_ARRAY(Functor=F.BPVIP(Vertices =pvs) , Relations = tagAlg_rels), 
        "Tr_T_BPVIPCHI2" : F.MAP_INPUT_ARRAY(Functor=  F.BPVIPCHI2(pvs) , Relations = tagAlg_rels), 
        "Tr_T_Charge": F.VALUE_OR(int_class) @ F.MAP_INPUT_ARRAY(Functor= F.CHARGE, Relations=tagAlg_rels), 
        "Tr_T_ISMUON": F.VALUE_OR(int_class) @ F.MAP_INPUT_ARRAY(Functor= F.ISMUON, Relations=tagAlg_rels),
        "Tr_T_ENERGY": F.MAP_INPUT_ARRAY(Functor= F.ENERGY, Relations=tagAlg_rels),
        "Tr_T_Eta": F.MAP_INPUT_ARRAY(Functor= F.ETA, Relations=tagAlg_rels),
        "Tr_T_MINIP": F.MAP_INPUT_ARRAY(Functor=F.MINIP(Vertices =pvs), Relations=tagAlg_rels),
        "Tr_T_MINIPChi2": F.MAP_INPUT_ARRAY(Functor= F.MINIPCHI2(Vertices =pvs), Relations=tagAlg_rels), 
        "Tr_T_P": F.MAP_INPUT_ARRAY(Functor=F.P, Relations=tagAlg_rels),
        "Tr_T_PT": F.MAP_INPUT_ARRAY(Functor= F.PT, Relations=tagAlg_rels),
        "Tr_T_PIDK": F.MAP_INPUT_ARRAY(Functor= F.PID_K, Relations=tagAlg_rels),
        "Tr_T_PIDe": F.MAP_INPUT_ARRAY(Functor= F.PID_E, Relations=tagAlg_rels),
        "Tr_T_PIDmu": F.MAP_INPUT_ARRAY(Functor= F.PID_MU, Relations=tagAlg_rels),
        "Tr_T_PIDP": F.MAP_INPUT_ARRAY(Functor= F.PID_P, Relations=tagAlg_rels),
        "Tr_T_PROBNN_GHOST": F.MAP_INPUT_ARRAY(Functor= F.PROBNN_GHOST, Relations=tagAlg_rels),
        "Tr_T_PROBNN_E": F.MAP_INPUT_ARRAY(Functor= F.PROBNN_E, Relations=tagAlg_rels),
        "Tr_T_PROBNN_K": F.MAP_INPUT_ARRAY(Functor= F.PROBNN_K, Relations=tagAlg_rels),
        "Tr_T_PROBNN_P": F.MAP_INPUT_ARRAY(Functor= F.PROBNN_P, Relations=tagAlg_rels),
        "Tr_T_PROBNN_MU": F.MAP_INPUT_ARRAY(Functor= F.PROBNN_MU, Relations=tagAlg_rels),
        "Tr_T_PROBNN_PI": F.MAP_INPUT_ARRAY(Functor= F.PROBNN_PI, Relations=tagAlg_rels),
        "Tr_T_BPVX": F.MAP_INPUT_ARRAY(Functor= F.BPVX(pvs), Relations=tagAlg_rels),
        "Tr_T_BPVY": F.MAP_INPUT_ARRAY(Functor= F.BPVY(pvs), Relations=tagAlg_rels),
        "Tr_T_BPVZ": F.MAP_INPUT_ARRAY(Functor= F.BPVZ(pvs), Relations=tagAlg_rels),
        "Tr_T_Phi": F.MAP_INPUT_ARRAY(Functor= F.PHI, Relations = tagAlg_rels),
        #"Tr_T_samePVasB": F.MAP_INPUT_ARRAY(Functor= F.SHARE_BPV(pvs), Relations = tagAlg_rels),
        "Tr_T_M": F.MAP_INPUT_ARRAY(Functor= F.MASS, Relations = tagAlg_rels),
        "Tr_T_CHI2DOF" :  F.MAP_INPUT_ARRAY(Functor= (F.CHI2DOF), Relations = tagAlg_rels), 
        "Tr_T_GHOSTPROB": F.MAP_INPUT_ARRAY(Functor=F.GHOSTPROB, Relations = tagAlg_rels),
        "Tr_T_PX" : F.MAP_INPUT_ARRAY(Functor= F.PX, Relations = tagAlg_rels),
        "Tr_T_PY" : F.MAP_INPUT_ARRAY(Functor= F.PY, Relations = tagAlg_rels),
        "Tr_T_PZ" : F.MAP_INPUT_ARRAY(Functor= F.PZ, Relations = tagAlg_rels),
        "Tr_T_TRUEID":  F.VALUE_OR(0) @ F.MAP_INPUT_ARRAY(Functor=F.VALUE_OR(0) @ MCTRUTH_TAG(F.PARTICLE_ID),Relations=tagAlg_rels),
        "Tr_T_X": F.MAP_INPUT_ARRAY(Functor= F.REFERENCEPOINT_X, Relations=tagAlg_rels),
        "Tr_T_Y": F.MAP_INPUT_ARRAY(Functor= F.REFERENCEPOINT_Y, Relations=tagAlg_rels),
        "Tr_T_Z": F.MAP_INPUT_ARRAY(Functor= F.REFERENCEPOINT_Z, Relations=tagAlg_rels),
        "ENDVX":F.END_VX,
        "ENDVY":F.END_VY,
        "ENDVZ":F.END_VZ,
        "Tr_T_TRUEPRIMARYVERTEX_X": F.VALUE_OR(-1000) @ F.MAP_INPUT_ARRAY(Functor = F.VALUE_OR(-1000) @ MCTRUTH_TAG(F.MC_PV_VX), Relations = tagAlg_rels),
        "Tr_T_TRUEPRIMARYVERTEX_Y": F.VALUE_OR(-1000) @ F.MAP_INPUT_ARRAY(Functor = F.VALUE_OR(-1000) @ MCTRUTH_TAG(F.MC_PV_VY), Relations = tagAlg_rels),
        "Tr_T_TRUEPRIMARYVERTEX_Z": F.VALUE_OR(-1000) @ F.MAP_INPUT_ARRAY(Functor = F.VALUE_OR(-1000) @ MCTRUTH_TAG(F.MC_PV_VZ), Relations = tagAlg_rels),
        "Tr_T_TRUEORIGINVERTEX_X": F.VALUE_OR(-1000) @ F.MAP_INPUT_ARRAY(Functor = F.VALUE_OR(-1000) @ MCTRUTH_TAG(F.ORIGIN_VX), Relations = tagAlg_rels),
        "Tr_T_TRUEORIGINVERTEX_Y": F.VALUE_OR(-1000) @ F.MAP_INPUT_ARRAY(Functor = F.VALUE_OR(-1000) @ MCTRUTH_TAG(F.ORIGIN_VY), Relations = tagAlg_rels),
        "Tr_T_TRUEORIGINVERTEX_Z": F.VALUE_OR(-1000) @ F.MAP_INPUT_ARRAY(Functor = F.VALUE_OR(-1000) @ MCTRUTH_TAG(F.ORIGIN_VZ), Relations = tagAlg_rels),
        "Tr_T_MC_MOTHER_ID": MCMOTHER_ID(1),
        "Tr_T_MC_MOTHER_KEY": MCMOTHER_KEY(1),
        "Tr_T_MC_GD_MOTHER_ID": MCMOTHER_ID(2),
        "Tr_T_MC_GD_MOTHER_KEY": MCMOTHER_KEY(2),
        "Tr_T_MC_GD_GD_MOTHER_ID": MCMOTHER_ID(3),
        "Tr_T_MC_GD_GD_MOTHER_KEY": MCMOTHER_KEY(3),

    })

    mckin = MCKinematics(mctruth_alg=mctruth)
    mchierarchy = MCHierarchy(mctruth_alg=mctruth) 

    bkg_cat = FunctorCollection({"BKGCAT": F.BKGCAT(Relations=mctruth.BkgCatTable)})
    variables = {
        "ALL" : variables_all + all_variables_add + mckin + mchierarchy ,
        "B" : variables_Bu + bkg_cat,
    }
   
    ODIN = get_odin()
    evt_vars = EventInfo()
    evt_vars.update({"EVENTTYPE": F.EVENTTYPE(ODIN)})
    
    tuple_B2JpsiK = Funtuple(
    name="Tuple",
    tuple_name="DecayTree",
    fields=fields,
    variables=variables,
    event_variables=evt_vars,
    inputs=B2JpsiK)
    return(make_config(options , [require_pvs(pvs), tuple_B2JpsiK]))