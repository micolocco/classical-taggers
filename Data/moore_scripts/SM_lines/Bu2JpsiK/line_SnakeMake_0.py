from Moore import options, run_moore
from Hlt2Conf.lines.hlt2_line_B_JpsiK_FT import Bu2JpsiK_Jpsi2MuMu_line

from RecoConf.global_tools import stateProvider_with_simplified_geom
from Gaudi.Configuration import FileCatalog
from RecoConf.reconstruction_objects import (
    reconstruction,
    #make_charged_protoparticles,
    #make_pvs, upfront_reconstruction
)
from PyConf.application import configure_input, configure
import os 
from RecoConf.hlt1_muonid import make_muon_hits

def all_lines():
    return [Bu2JpsiK_Jpsi2MuMu_line()]


public_tools  = [stateProvider_with_simplified_geom()]

catalog_sample = "MC_Upgrade_12143001_Beam7000GeVUpgradeMagDownNu7.625nsPythia8_Sim10aU1_XDIGI.py.xml"

LFN_path = f"/ceph/users/molocco/classical-taggers/Data/moore_scripts/LFNs/Bu2JpsiK/"
LFN_list = open(f"{LFN_path}LFN_0.txt","r").read().split("\n")

FileCatalog().Catalogs = [f'xmlcatalog_file:/ceph/users/molocco/classical-taggers/Data/Database/Bu2JpsiK/{catalog_sample}' ]
options.input_files =  LFN_list

outputPath = f"/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/output"
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

options.input_raw_format = 0.5 # 4.3
options.evt_max = -1
options.simulation = True
options.print_freq = 1000
options.input_type = 'ROOT'
options.output_file = f'{outputPath}/hlt2_SM_0.dst'
options.output_type = 'ROOT'
options.dddb_tag = 'dddb-20210617'  
options.conddb_tag = 'sim-20210617-vc-md100' 
options.output_manifest_file = f'{outputPath}/hlt2_tck_SM_0.json'

make_muon_hits.global_bind(geometry_version=2)

with reconstruction.bind(from_file=False):
    run_moore(options, all_lines, public_tools)




