# scripts/getOptimized.py
import os, json, re, glob, argparse, yaml, numpy as np
from itertools import product
import utils
from scripts.replace_path import seeds  # your seeds

"""
Example:
python scripts/getOptimized.py \
  --model_prePath /ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024 \
  --cut allBKGCAT_notSamePV_noOSP_SSK_balanced \
  --features union_PROBNN \
  --asym asym_level1 \
  --outputPath /home/molocco/classical-taggers/best_tagger_candidates
"""

def tp_nominal_sigma(tp):
    """Return (nominal, sigma) from either an uncertainties ufloat or a dict."""
    try:
        return tp.nominal_value, tp.std_dev
    except AttributeError:
        # Expect dict-like {"nominal_value": ..., "std_dev": ...}
        return tp.get("nominal_value", np.nan), tp.get("std_dev", np.nan)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Pick best model per tagger by maximizing TP_cali / sigma across all calibrations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_prePath', type=str, required=True,
                   help='…/savedModels/<sample_type> (root under which decays/taggers live)')
    p.add_argument('--cut', type=str, required=True,
                   help='cut folder, e.g. allBKGCAT_notSamePV_noOSP_SSK_balanced')
    p.add_argument('--features', type=str, default='union_PROBNN')
    p.add_argument('--asym', type=str, default='asym_level1')
    p.add_argument('--outputPath', type=str, default='/home/molocco/classical-taggers/best_tagger_candidates')
    p.add_argument('--outfile', type=str, default=None,
                   help='Optional explicit output file path for the candidates JSON.')
    args = p.parse_args()

    print("Config:", vars(args))

    # Which decay each tagger lives under
    tagger_dict = {
        "OSKaon":    "Bu2JpsiK",
        "OSElectron":"Bu2JpsiK",
        "OSMuon":    "Bu2JpsiK",
        "SSPion":    "Bd2JpsiKst",
        "SSProton":  "Bd2JpsiKst",
        "SSKaon":    "Bs2DsPi",
    }

    # hyperparameter grid
    with open('configs/hyperpar_intervals.yaml', 'r') as file:
        intervals = yaml.safe_load(file)
    learning_rates    = intervals['-learning_rate']
    train_batch_sizes = intervals['-train_batch_size']
    numlayers         = intervals['-numlayers']
    numneurons        = intervals['-numneurons']

    combos = list(product(seeds, learning_rates, train_batch_sizes, numlayers, numneurons))

    results = {}
    # regex helpers
    re_npar = re.compile(r'calibration_npar(\d+)')
    re_func = re.compile(r'taggingInfo_([A-Za-z0-9_]+)\.json$')

    for tagger, decay in tagger_dict.items():
        best = None
        best_nominal = -np.inf
        n_found = 0

        for seed, lr, bs, nL, nN in combos:
            # trial dir = .../<seed> / lr..._bs..._nL..._nN... / <asym>
            trial_dir = os.path.join(
                args.model_prePath, decay, tagger, args.cut, args.features,
                str(seed), f"lr{lr}_bs{bs}_nL{nL}_nN{nN}", args.asym
            )
            if not os.path.isdir(trial_dir):
                continue

            # scan all calibrations under this trial:
            pattern = os.path.join(trial_dir, "calibration_npar*", "*", "taggingInfo_*.json")
            for jf in glob.glob(pattern):
                m_npar = re_npar.search(jf)
                m_func = re_func.search(jf)
                if not (m_npar and m_func):
                    continue
                npar = int(m_npar.group(1))
                func = m_func.group(1)

                try:
                    data = utils.load_and_process_json(jf)
                except Exception as e:
                    print(f"[warn] failed to load {jf}: {e}")
                    continue

                tp = data.get('TaggingPower_Cali', None)
                if tp is None:
                    continue

                nominal, sigma = tp_nominal_sigma(tp)
                if (nominal is None) or (sigma is None):
                    continue
                if np.isnan(nominal) or np.isnan(sigma) or nominal == 0 or sigma == 0:
                    continue

                ratio = float(nominal) / float(sigma)
                precision = float(sigma) / float(nominal)

                if nominal > best_nominal:
                    best_nominal = nominal
                    results[tagger] = {
                        "calibrated tagging power": f"{tp}",
                        "seed": seed,
                        "learning_rate": lr,
                        "batch_size": bs,
                        "numlayers": nL,
                        "numneurons": nN,
                        "npar": npar,
                        "function": func,
                        "max_ratio": ratio,
                        "precision": precision,
                        "model_dir": trial_dir       # points to .../<asym>
                    }
                n_found += 1
        print("Found", n_found, "valid calibrations for tagger", tagger)

    # write output
    if args.outfile:
        out_file = args.outfile
    else:
        out_file = os.path.join(args.outputPath, args.cut, args.features, args.asym, "candidatedTaggers_overall_noTPError.json")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[getOptimized] wrote {out_file}")
