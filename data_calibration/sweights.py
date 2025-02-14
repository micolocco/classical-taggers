import uproot
import argparse
import datetime
import os

from scripts.adding_features_v2 import loading_variables, data_vars_translation

_loading_variables = []
_loading_variables.append("B_Tr_T_IsInTree")
_loading_variables.append("B_ID")
_loading_variables.append("FillNumber")
for v in loading_variables:
    if "TRUE" in v or "BKGCAT" in v or "Origin_Flag" in v or "MC" in v: continue
    if "BPV" in v: v=v.replace("BPV", "OWNPV_").replace("OWNPV_IP", "OWNPVIP")
    if "END_V" in v: v=v.replace("END_V", "ENDV_")
    _loading_variables.append(v) if v not in data_vars_translation.keys() else _loading_variables.append(data_vars_translation[v])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the tagger on the specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', help='File to be read from eos')
    parser.add_argument('--treename')
    parser.add_argument('--output')
    


cfg = parser.parse_args()
file = cfg.input
print(f'Start reading: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
with uproot.open("{}".format(file)) as _f:
    _df = _f[cfg.treename].arrays(_loading_variables, library="pd")

# Equivalent for data of Origin_Flag != 0
_df.drop(_df[_df['B_Tr_T_IsInTree'] != 1 ].index , inplace = True)

import ipdb; ipdb.set_trace()
os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
with uproot.recreate(cfg.output) as f:
    f[cfg.treename] = _df

print(f'Modified NTuple processed and saved to {cfg.output}')
print(f'Creation time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')