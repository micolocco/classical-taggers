#!/usr/bin/env python3

"""Compare the tagging powers of all common taggers in two combinations."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from sandwich_correlation import calculate_tagging_power_correlation


TAGGER_ORDER = (
    'OSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton', 'SSKaon'
)
TAGGER_LABELS = {'OSElectron': 'OSElec', 'SSProton': 'SSProt'}


def _load_calibration(path):
    with open(path) as calibration_file:
        return json.load(calibration_file)


def _taggers_with_power(calibration):
    """Return calibration entries that contain an overall tagging power."""
    taggers = []
    for name, values in calibration.items():
        try:
            power = values['calibrated']['overall']['tagging_power']
        except (KeyError, TypeError):
            continue
        if isinstance(power, (list, tuple)) and len(power) == 3:
            taggers.append(name)
    return taggers


def _combined_tagger_name(calibration):
    """Return the combined-tagger entry from an ftcalib calibration mapping."""
    candidates = _taggers_with_power(calibration)
    return max(candidates, key=lambda name: (name.count('_'), len(name)))


def _read_tagging_power(calibration, tagger, source=None):
    power = calibration[tagger]['calibrated']['overall']['tagging_power']
    power = np.asarray(power, dtype=float)
    if np.isnan(power[1]):
        location = f' in {source}' if source else ''
        print(
            f'WARNING: statistical uncertainty for {tagger}{location} is NaN; '
            'using 0.',
            file=sys.stderr,
            flush=True,
        )
        power[1] = 0.0
    return power


def read_tagging_power(path, tagger=None):
    """Read ``(value, stat_unc, calibration_unc)`` from a calibration JSON."""
    calibration = _load_calibration(path)
    tagger = tagger or _combined_tagger_name(calibration)
    return tagger, _read_tagging_power(calibration, tagger, source=path)


def _default_labels(paths):
    """Choose concise labels from the nearest differing path components."""
    first, second = paths
    first_name = os.path.basename(first)
    second_name = os.path.basename(second)
    if first_name != second_name:
        return [
            os.path.splitext(first_name)[0],
            os.path.splitext(second_name)[0],
        ]
    for part1, part2 in zip(
        reversed(os.path.normpath(os.path.dirname(first)).split(os.sep)),
        reversed(os.path.normpath(os.path.dirname(second)).split(os.sep)),
    ):
        if part1 != part2:
            return [part1, part2]
    return ['calibration 1', 'calibration 2']


def mc_path(path):
    """Return the corresponding MC path, if ``FlavourTagging/Data`` is present."""
    data_directory = os.path.join('FlavourTagging', 'Data')
    mc_directory = os.path.join('FlavourTagging', 'MC')
    return path.replace(data_directory, mc_directory, 1)


def propagated_difference(power1, power2, calibration_correlation):
    """Return value and Gaussian uncertainties for ``power1 - power2``.

    Statistical uncertainties are treated as 100% correlated. Calibration
    uncertainties use the supplied sandwich-estimator correlation.
    """
    value1, stat1, calibration1 = np.asarray(power1, dtype=float)
    value2, stat2, calibration2 = np.asarray(power2, dtype=float)
    stat_variance = stat1**2 + stat2**2 - 2.0 * stat1 * stat2
    calibration_variance = (
        calibration1**2
        + calibration2**2
        - 2.0 * calibration_correlation * calibration1 * calibration2
    )
    stat_unc = np.sqrt(max(0.0, stat_variance))
    calibration_unc = np.sqrt(max(0.0, calibration_variance))
    return np.array([
        value1 - value2,
        stat_unc,
        calibration_unc,
        np.hypot(stat_unc, calibration_unc),
    ])


def _format_power(power, digits):
    """Format fractional power as percent: value +/- calibration +/- stat."""
    value, stat_unc, calibration_unc = 100.0 * np.asarray(power, dtype=float)
    return (
        f'{value:.{digits}f} ± {calibration_unc:.{digits}f} '
        f'± {stat_unc:.{digits}f}'
    )


def _format_difference(difference, digits):
    """Format a fractional difference and its total uncertainty in percent."""
    value, _, _, total_unc = 100.0 * np.asarray(difference, dtype=float)
    return f'{value:.{digits}f} ± {total_unc:.{digits}f}'


def _constituent_taggers(combined_name, available):
    """Return constituents in combination order, limited to real JSON entries."""
    return [name for name in combined_name.split('_') if name in available]


def _ordered_taggers(taggers):
    """Use the conventional display order, retaining any unfamiliar taggers."""
    taggers = list(dict.fromkeys(taggers))
    return (
        [name for name in TAGGER_ORDER if name in taggers]
        + [name for name in taggers if name not in TAGGER_ORDER]
    )


def _row_specs(calibrations, combined_names, constituent_taggers=None):
    """Return rows for taggers present in both supplied calibrations."""
    available = [set(_taggers_with_power(item)) for item in calibrations]
    if constituent_taggers is None:
        taggers = _constituent_taggers(combined_names[0], available[0])
    else:
        taggers = list(constituent_taggers)
    common_taggers = [
        name for name in _ordered_taggers(taggers)
        if name in available[0] and name in available[1]
    ]
    rows = [
        (TAGGER_LABELS.get(name, name), name, [name])
        for name in common_taggers
    ]

    if combined_names[0] == combined_names[1] and all(
        combined_names[0] in names for names in available
    ):
        combined_constituents = _constituent_taggers(
            combined_names[0], available[0]
        )
        if len(combined_constituents) > 1:
            rows.append(('comb', combined_names[0], combined_constituents))
    return rows


def _comparison_table(
    calibration_files,
    labels,
    *,
    tagger,
    constituent_taggers,
    mode,
    data_type,
    tree,
    digits,
):
    """Build one Data or MC table and calculate its correlations."""
    columns = pd.MultiIndex.from_tuples([
        (labels[0], f'tagging power on {data_type}'),
        (labels[1], f'tagging power on {data_type}'),
        (r'\Delta', f'{labels[0]} - {labels[1]}'),
        (r'\sigma_{\Delta}', ''),
    ])
    rows = []
    row_names = []
    correlations = {}
    calibrations = [_load_calibration(path) for path in calibration_files]
    combined_names = [
        tagger or _combined_tagger_name(calibration)
        for calibration in calibrations
    ]
    row_specs = _row_specs(
        calibrations, combined_names, constituent_taggers
    )

    for display_name, calibration_name, correlation_taggers in row_specs:
        powers = [
            _read_tagging_power(calibration, calibration_name, source=path)
            for calibration, path in zip(calibrations, calibration_files)
        ]
        correlation_matrix = calculate_tagging_power_correlation(
            calibration_files,
            correlation_taggers,
            mode=mode,
            data_type=data_type,
            tree=tree,
        )
        correlation = float(correlation_matrix[0, 1])
        correlations[display_name] = correlation
        difference = propagated_difference(powers[0], powers[1], correlation)
        significance = (
            difference[0] / difference[3]
            if difference[3] > 0.0
            else np.nan
        )
        rows.append([
            _format_power(powers[0], digits),
            _format_power(powers[1], digits),
            _format_difference(difference, digits),
            significance,
        ])
        row_names.append(display_name)

    table = pd.DataFrame(rows, index=row_names, columns=columns)
    table.index.name = 'tagger'
    return table, correlations


def compare_calibrations(
    calibration_files,
    *,
    labels=None,
    tagger=None,
    constituent_taggers=None,
    mode='Bd',
    tree='DecayTree;1',
    digits=4,
):
    """Return separate Data/MC tables and per-tagger correlations.

    Only constituent taggers found in both combinations are included. A
    combined (``comb``) row is added when both JSONs contain the same combined
    tagger entry.
    """

    labels = labels or _default_labels(calibration_files)
    print(
        f'Loading Data calibrations from {calibration_files[0]} '
        f'and {calibration_files[1]}...', flush=True
    )
    data_table, data_correlations = _comparison_table(
        calibration_files,
        labels,
        tagger=tagger,
        constituent_taggers=constituent_taggers,
        mode=mode,
        data_type='Data',
        tree=tree,
        digits=digits,
    )
    tables = {'Data': data_table}
    correlations = {'Data': data_correlations}

    mc_files = [mc_path(path) for path in calibration_files]
    print(f"Loading MC calibrations from {mc_files[0]} and {mc_files[1]}...", flush=True)

    if all(os.path.exists(path) for path in mc_files):
        mc_table, mc_correlations = _comparison_table(
            mc_files,
            labels,
            tagger=tagger,
            constituent_taggers=constituent_taggers,
            mode='Bu',
            data_type='MC',
            tree=tree,
            digits=digits,
        )
        tables['MC'] = mc_table
        correlations['MC'] = mc_correlations

    return (
        tables,
        correlations,
    )


def main():
    digits = 3

    parser = argparse.ArgumentParser(
        description=(
            'Compare all common taggers in two combined calibration JSONs. '
            'Corresponding MC results are shown when available.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('calibrations', nargs=2, help='combined calibration JSONs')
    parser.add_argument('--labels', nargs=2, help='labels used in the output table')
    parser.add_argument(
        '--tagger',
        help='combined tagger key (normally inferred; must be common to both JSONs)',
    )
    parser.add_argument(
        '--constituent-taggers', nargs='+',
        help='candidate constituent taggers (missing ones are omitted)',
    )
    parser.add_argument('--mode', default='Bd', help='decay mode passed to ftcalib')
    parser.add_argument('--tree', default='DecayTree;1', help='input ROOT tree')
    args = parser.parse_args()

    tables, correlations = compare_calibrations(
        args.calibrations,
        labels=args.labels,
        tagger=args.tagger,
        constituent_taggers=args.constituent_taggers,
        mode=args.mode,
        tree=args.tree,
        digits=digits,
    )

    print(
        'Tagging-power entries: value ± calibration uncertainty ± '
        'statistical uncertainty [%]', flush=True
    )
    print('Differences: value ± total uncertainty [%]', flush=True)
    for data_type in tables.keys():
        print(f'\n{data_type} calibration-uncertainty correlations:', flush=True)
        for tagger_name, correlation in correlations[data_type].items():
            print(f'  {tagger_name}: {correlation:.{digits}f}', flush=True)
        print(f'\nPerformance on {data_type}:', flush=True)
        print(tables[data_type].to_string(
            float_format=lambda value: f'{value:.{digits}f}'
        ), flush=True)


if __name__ == '__main__':
    main()


'''
Command library


#Bd2JpsiKst combination v1_tree vs 2StageDT comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/v1_2StageSS_newFeatures_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
--labels 2StageDT v1_tree \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/v1DT_vs_2StageDT/Bd2JpsiKst_comb.log

#Bs2DsPi combination v1_tree vs 2StageDT comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/trained_Data/v1_2StageSS_newFeatures_balanced/union_PROBNN/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
--labels 2StageDT v1_tree \
--mode Bs \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/v1DT_vs_2StageDT/Bs2DsPi_comb.log

#Bu2JpsiK combination v1_tree vs 2StageDT comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/v1_2StageSS_newFeatures_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/calibration.json \
--labels 2StageDT v1_tree \
--mode Bu \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/v1DT_vs_2StageDT/Bu2JpsiK_comb.log



#Bd2JpsiKst combination 2StageDT_trained_Data vs 2StageDT_trained_MC comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/v1_2StageSS_newFeatures_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/v1_2StageSS_newFeatures_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
--labels Data_trained MC_trained \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/2StageDT_Data_vs_MC/Bd2JpsiKst_comb.log

#Bs2DsPi combination v1_tree_retrained_Data vs v1_taggers comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/trained_Data/v1_2StageSS_newFeatures_balanced/union_PROBNN/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/trained_MC/v1_2StageSS_newFeatures_balanced/union_PROBNN/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
--labels Data_trained MC_trained \
--mode Bs \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/2StageDT_Data_vs_MC/Bs2DsPi_comb.log

#Bu2JpsiK combination v1_tree_retrained_Data vs v1_taggers comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/v1_2StageSS_newFeatures_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_MC/v1_2StageSS_newFeatures_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/calibration.json \
--labels Data_trained MC_trained \
--mode Bu \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/2StageDT_Data_vs_MC/Bu2JpsiK_comb.log



#Bd2JpsiKst combination v1_tree_retrained_Data vs v1_tree_retrained_MC comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
--labels Data_trained MC_trained \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/Data_vs_MC/Bd2JpsiKst_comb.log

#Bs2DsPi combination v1_tree_retrained_Data vs v1_taggers comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
--labels Data_trained MC_trained \
--mode Bs \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/Data_vs_MC/Bs2DsPi_comb.log

#Bu2JpsiK combination v1_tree_retrained_Data vs v1_taggers comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/calibration.json \
--labels Data_trained MC_trained \
--mode Bu \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/Data_vs_MC/Bu2JpsiK_comb.log



#Bd2JpsiKst combination v1_tree_retrained_Data vs v1_taggerscomparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/old_training_allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
--labels v1_tree_retrained v1_taggers \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/Data_vs_benchmark/Bd2JpsiKst_comb.log

#Bs2DsPi combination v1_tree_retrained_Data vs v1_taggers comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/Run3v1/old_training_allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
--labels v1_tree_retrained v1_taggers \
--mode Bs \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/Data_vs_benchmark/Bs2DsPi_comb.log

#Bu2JpsiK combination v1_tree_retrained_Data vs v1_taggers comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/Run3v1/old_training_allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/OSKaon_OSMuon_OSElectron/calibration.json \
--labels v1_tree_retrained v1_taggers \
--mode Bu \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/Data_vs_benchmark/Bu2JpsiK_comb.log



#Bd2JpsiKst combination v1_tree_retrained_MC vs v1_taggerscomparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/calibration.json \
--labels v1_tree_retrained v1_taggers \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/MC_vs_benchmark/Bd2JpsiKst_comb.log

#Bs2DsPi combination v1_tree_retrained_MC vs v1_taggers comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2DsPi/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSKaon_OSKaon_OSMuon_OSElectron/calibration.json \
--labels v1_tree_retrained v1_taggers \
--mode Bs \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/MC_vs_benchmark/Bs2DsPi_comb.log

#Bu2JpsiK combination v1_tree_retrained_MC vs v1_taggers comparison:
python scripts/tagging_power_comp_table.py \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/calibration.json \
/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/OSKaon_OSMuon_OSElectron/calibration.json \
--labels v1_tree_retrained v1_taggers \
--mode Bu \
&> /ceph/users/togasa/FlavourTagging/comparisons/tagging_power_diff/MC_vs_benchmark/Bu2JpsiK_comb.log





'''
