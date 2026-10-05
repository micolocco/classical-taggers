"""Numba-accelerated, progressive n-body reconstruction for dataframes.

Each input row is used as the seed of a reconstruction hypothesis. The module
first combines that track with one other track, then repeatedly combines the
resulting composite with one unused track from the same ``candidate_entry``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from numba import njit, prange
from tqdm.auto import tqdm


__all__ = [
    "add_n_body_candidate_features",
    "add_signal_b_daughter_mass_features",
    "add_track_signal_b_poca_features",
    "add_two_body_candidate_features",
    "kaon_mass_hypothesis",
    "zero_uncombined_mass",
]

CANDIDATE_COLUMN = "candidate_entry"
# Coordinates are in mm; momenta and the default mass hypothesis are in MeV.
POSITION_COLUMNS = ("B_Tr_T_X", "B_Tr_T_Y", "B_Tr_T_Z")
MOMENTUM_COLUMNS = ("B_Tr_T_PX", "B_Tr_T_PY", "B_Tr_T_PZ")
PV_COLUMNS = ("B_OWNPV_X", "B_OWNPV_Y", "B_OWNPV_Z")
SIGNAL_B_POSITION_COLUMNS = ("B_ENDV_X", "B_ENDV_Y", "B_ENDV_Z")
SIGNAL_B_MOMENTUM_COLUMNS = ("B_PX", "B_PY", "B_PZ")
MASS_HYPOTHESIS_COLUMN = "B_Tr_T_mass_hypothesis"

KAON_MASS_MEV = 493.677
TrackValueFunction = Callable[[pd.DataFrame], object]


def kaon_mass_hypothesis(track_variables: pd.DataFrame) -> np.ndarray:
    """Return a charged-kaon mass hypothesis in MeV/c^2 for every track."""
    return np.full(len(track_variables), KAON_MASS_MEV, dtype=np.float64)


def zero_uncombined_mass(track_variables: pd.DataFrame) -> np.ndarray:
    """Return the invariant-mass value used when a row cannot be combined."""
    return np.zeros(len(track_variables), dtype=np.float64)


@njit(cache=True, parallel=True)
def _track_signal_b_poca_batch(
    track_positions: np.ndarray,
    track_directions: np.ndarray,
    signal_b_positions: np.ndarray,
    signal_b_directions: np.ndarray,
    first_row: int,
    last_row: int,
    doca: np.ndarray,
    poca: np.ndarray,
) -> None:
    """Fit each row's track line against its signal-B line."""
    for row in prange(first_row, last_row):
        u1x = track_directions[row, 0]
        u1y = track_directions[row, 1]
        u1z = track_directions[row, 2]
        u2x = signal_b_directions[row, 0]
        u2y = signal_b_directions[row, 1]
        u2z = signal_b_directions[row, 2]
        if not np.isfinite(u1x) or not np.isfinite(u2x):
            continue

        # Lines are track_position + t1*track_direction and
        # B_ENDV + t2*B_direction. Parallel lines have no unique POCA.
        direction_dot = u1x * u2x + u1y * u2y + u1z * u2z
        denominator = 1.0 - direction_dot * direction_dot
        if denominator <= 1.0e-12:
            continue

        rx = signal_b_positions[row, 0] - track_positions[row, 0]
        ry = signal_b_positions[row, 1] - track_positions[row, 1]
        rz = signal_b_positions[row, 2] - track_positions[row, 2]
        r_dot_u1 = rx * u1x + ry * u1y + rz * u1z
        r_dot_u2 = rx * u2x + ry * u2y + rz * u2z
        t1 = (r_dot_u1 - direction_dot * r_dot_u2) / denominator
        t2 = (direction_dot * r_dot_u1 - r_dot_u2) / denominator

        closest_track_x = track_positions[row, 0] + t1 * u1x
        closest_track_y = track_positions[row, 1] + t1 * u1y
        closest_track_z = track_positions[row, 2] + t1 * u1z
        closest_b_x = signal_b_positions[row, 0] + t2 * u2x
        closest_b_y = signal_b_positions[row, 1] + t2 * u2y
        closest_b_z = signal_b_positions[row, 2] + t2 * u2z

        dx = closest_track_x - closest_b_x
        dy = closest_track_y - closest_b_y
        dz = closest_track_z - closest_b_z
        doca[row] = np.sqrt(dx * dx + dy * dy + dz * dz)
        poca[row, 0] = 0.5 * (closest_track_x + closest_b_x)
        poca[row, 1] = 0.5 * (closest_track_y + closest_b_y)
        poca[row, 2] = 0.5 * (closest_track_z + closest_b_z)


@njit(cache=True, parallel=True)
def _extend_combinations(
    track_positions: np.ndarray,
    track_momenta: np.ndarray,
    track_directions: np.ndarray,
    candidate_first: np.ndarray,
    candidate_last: np.ndarray,
    previous_poca: np.ndarray,
    previous_momentum: np.ndarray,
    previous_direction: np.ndarray,
    members: np.ndarray,
    number_of_members: int,
    maximum_doca: float,
    upstream_tolerance: float,
    first_row: int,
    last_row: int,
    added_track: np.ndarray,
    doca: np.ndarray,
    poca: np.ndarray,
    momentum: np.ndarray,
    vertex_distance: np.ndarray,
) -> None:
    """Add the closest unused track to every composite candidate in a row range."""
    # Rows are independent reconstruction hypotheses, which makes this loop
    # safe to parallelize. At the first stage, previous_* describes the seed
    # track; at later stages it describes the composite built one stage earlier.
    for row in prange(first_row, last_row):
        u1x = previous_direction[row, 0]
        u1y = previous_direction[row, 1]
        u1z = previous_direction[row, 2]
        if not np.isfinite(u1x):
            continue

        best_track = -1
        # Initializing with the cut makes the comparison below enforce the
        # strict requirement DOCA < maximum_doca.
        best_doca = maximum_doca
        best_poca_x = np.nan
        best_poca_y = np.nan
        best_poca_z = np.nan

        for track in range(candidate_first[row], candidate_last[row]):
            # A physical input track may occur only once in a combination.
            already_used = False
            for member_number in range(number_of_members):
                if members[row, member_number] == track:
                    already_used = True
                    break
            if already_used:
                continue

            u2x = track_directions[track, 0]
            u2y = track_directions[track, 1]
            u2z = track_directions[track, 2]
            if not np.isfinite(u2x):
                continue

            # Find the closest points on the infinite lines
            #   previous_poca + t1 * previous_direction
            #   track_position + t2 * track_direction.
            direction_dot = u1x * u2x + u1y * u2y + u1z * u2z
            denominator = 1.0 - direction_dot * direction_dot
            # Parallel tracks have no unique pair of closest points.
            if denominator <= 1.0e-12:
                continue

            rx = track_positions[track, 0] - previous_poca[row, 0]
            ry = track_positions[track, 1] - previous_poca[row, 1]
            rz = track_positions[track, 2] - previous_poca[row, 2]
            r_dot_u1 = rx * u1x + ry * u1y + rz * u1z
            r_dot_u2 = rx * u2x + ry * u2y + rz * u2z

            t1 = (r_dot_u1 - direction_dot * r_dot_u2) / denominator
            t2 = (direction_dot * r_dot_u1 - r_dot_u2) / denominator

            closest1x = previous_poca[row, 0] + t1 * u1x
            closest1y = previous_poca[row, 1] + t1 * u1y
            closest1z = previous_poca[row, 2] + t1 * u1z
            closest2x = track_positions[track, 0] + t2 * u2x
            closest2y = track_positions[track, 1] + t2 * u2y
            closest2z = track_positions[track, 2] + t2 * u2z

            dx = closest1x - closest2x
            dy = closest1y - closest2y
            dz = closest1z - closest2z
            track_doca = np.sqrt(dx * dx + dy * dy + dz * dz)

            candidate_poca_x = 0.5 * (closest1x + closest2x)
            candidate_poca_y = 0.5 * (closest1y + closest2y)
            candidate_poca_z = 0.5 * (closest1z + closest2z)

            # The first (2-body) fit defines the first reconstructed vertex.
            # At every later stage, the new parent vertex must be upstream of
            # the preceding vertex along the preceding composite's momentum.
            # A small positive displacement is allowed for vertex resolution.
            if number_of_members >= 2:
                longitudinal_displacement = (
                    (candidate_poca_x - previous_poca[row, 0]) * u1x
                    + (candidate_poca_y - previous_poca[row, 1]) * u1y
                    + (candidate_poca_z - previous_poca[row, 2]) * u1z
                )
                if longitudinal_displacement > upstream_tolerance:
                    continue

            # Retain only the closest track that also passes the DOCA cut.
            if track_doca < best_doca:
                best_track = track
                best_doca = track_doca
                best_poca_x = candidate_poca_x
                best_poca_y = candidate_poca_y
                best_poca_z = candidate_poca_z

        # Leave the preallocated default values unchanged if no track passed.
        if best_track >= 0:
            added_track[row] = best_track
            doca[row] = best_doca
            poca[row, 0] = best_poca_x
            poca[row, 1] = best_poca_y
            poca[row, 2] = best_poca_z
            momentum[row, 0] = previous_momentum[row, 0] + track_momenta[best_track, 0]
            momentum[row, 1] = previous_momentum[row, 1] + track_momenta[best_track, 1]
            momentum[row, 2] = previous_momentum[row, 2] + track_momenta[best_track, 2]
            # Separation of successive POCAs distinguishes a prompt common
            # vertex from a spatially separated decay-chain vertex.
            vx = best_poca_x - previous_poca[row, 0]
            vy = best_poca_y - previous_poca[row, 1]
            vz = best_poca_z - previous_poca[row, 2]
            vertex_distance[row] = np.sqrt(vx * vx + vy * vy + vz * vz)


def _batches(
    number_of_rows: int, batch_size: int, show_progress: bool, body_count: int
) -> Iterator[tuple[int, int]]:
    """Yield bounded row ranges and optionally display their progress."""
    # Several smaller Numba calls make progress observable without moving the
    # numerical inner loops back into Python.
    starts = range(0, number_of_rows, batch_size)
    if show_progress:
        starts = tqdm(
            starts,
            total=(number_of_rows + batch_size - 1) // batch_size,
            desc=f"{body_count}-body reconstruction",
            unit="batch",
            dynamic_ncols=True,
        )
    for start in starts:
        yield start, min(start + batch_size, number_of_rows)


def _directions(momentum: np.ndarray) -> np.ndarray:
    """Normalize an (N, 3) vector array, marking invalid directions as NaN."""
    norms = np.sqrt(np.sum(momentum * momentum, axis=1))
    directions = np.full_like(momentum, np.nan)
    valid = np.isfinite(norms) & (norms > 0.0)
    directions[valid] = momentum[valid] / norms[valid, np.newaxis]
    return directions


def add_track_signal_b_poca_features(
    df: pd.DataFrame,
    *,
    inplace: bool = True,
    show_progress: bool = True,
    row_batch_size: int = 200_000,
) -> pd.DataFrame:
    """Add the POCA and DOCA between each single track and the signal B.

    The track line is anchored at ``B_Tr_T_X/Y/Z`` and follows the track
    momentum. The signal-B line is anchored at ``B_ENDV_X/Y/Z`` and follows
    ``B_PX/PY/PZ``. The reported POCA is the midpoint of the closest point on
    each line, matching the convention used by the n-body reconstruction.

    No tracks are combined and no ``candidate_entry`` grouping is performed.
    Invalid zero-momentum or parallel line pairs receive ``NaN`` outputs.
    """
    required = {
        *POSITION_COLUMNS,
        *MOMENTUM_COLUMNS,
        *SIGNAL_B_POSITION_COLUMNS,
        *SIGNAL_B_MOMENTUM_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")
    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")

    result_df = df if inplace else df.copy()
    number_of_rows = len(result_df)
    track_positions = np.ascontiguousarray(
        result_df.loc[:, POSITION_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    )
    track_momentum = np.ascontiguousarray(
        result_df.loc[:, MOMENTUM_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    )
    signal_b_positions = np.ascontiguousarray(
        result_df.loc[:, SIGNAL_B_POSITION_COLUMNS].to_numpy(
            dtype=np.float64, copy=False
        )
    )
    signal_b_momentum = np.ascontiguousarray(
        result_df.loc[:, SIGNAL_B_MOMENTUM_COLUMNS].to_numpy(
            dtype=np.float64, copy=False
        )
    )

    doca = np.full(number_of_rows, np.nan, dtype=np.float64)
    poca = np.full((number_of_rows, 3), np.nan, dtype=np.float64)
    batch_starts = range(0, number_of_rows, row_batch_size)
    if show_progress:
        batch_starts = tqdm(
            batch_starts,
            total=(number_of_rows + row_batch_size - 1) // row_batch_size,
            desc="Track-signal-B POCA",
            unit="batch",
            dynamic_ncols=True,
        )

    track_directions = _directions(track_momentum)
    signal_b_directions = _directions(signal_b_momentum)
    for first_row in batch_starts:
        _track_signal_b_poca_batch(
            track_positions,
            track_directions,
            signal_b_positions,
            signal_b_directions,
            first_row,
            min(first_row + row_batch_size, number_of_rows),
            doca,
            poca,
        )

    result_df["B_Tr_T_B_DOCA"] = doca
    result_df["B_Tr_T_B_POCA_X"] = poca[:, 0]
    result_df["B_Tr_T_B_POCA_Y"] = poca[:, 1]
    result_df["B_Tr_T_B_POCA_Z"] = poca[:, 2]
    return result_df


def add_signal_b_daughter_mass_features(
    df: pd.DataFrame,
    daughter_base_names: Sequence[str],
    *,
    daughter_mass_hypotheses: Mapping[str, float] | None = None,
    output_prefix: str = "signal_B_daughters",
    inplace: bool = True,
) -> pd.DataFrame:
    """Reconstruct a signal B from a chosen subset of its daughter tracks.

    ``daughter_base_names`` contains only the part of each column name before
    its kinematic suffix. For example, ``["muplus", "muminus", "hplus"]``
    reads ``muplus_PX``, ``muplus_PY``, ``muplus_PZ``, and likewise for the
    other daughters. Omit a base name to emulate a missing daughter.

    A daughter's energy is obtained, in order of preference, from an explicit
    entry in ``daughter_mass_hypotheses``, ``<base>_ENERGY``, ``<base>_PE``, or
    ``<base>_M``. A mass hypothesis is converted to energy using
    ``sqrt(PX**2 + PY**2 + PZ**2 + mass**2)``. This makes it possible to use
    tuples that retain only daughter momenta.

    The corrected mass uses the signal B flight direction
    ``B_ENDV - B_OWNPV`` and the summed momentum of the supplied visible
    daughters. Column names are constructed from ``output_prefix`` so several
    different missing-daughter hypotheses can coexist in one dataframe.

    The following columns are added:

    - ``<output_prefix>_energy``
    - ``<output_prefix>_PX/PY/PZ``
    - ``<output_prefix>_invariant_mass``
    - ``<output_prefix>_missing_PT``
    - ``<output_prefix>_corrected_mass``

    Rows with a non-physical four-vector or undefined flight direction receive
    ``NaN`` for the affected output.
    """
    daughter_base_names = tuple(daughter_base_names)
    if not daughter_base_names:
        raise ValueError("daughter_base_names must contain at least one daughter")
    if any(not isinstance(name, str) or not name for name in daughter_base_names):
        raise ValueError("every daughter base name must be a non-empty string")
    if len(set(daughter_base_names)) != len(daughter_base_names):
        raise ValueError("daughter_base_names must not contain duplicates")
    if not isinstance(output_prefix, str) or not output_prefix:
        raise ValueError("output_prefix must be a non-empty string")

    mass_hypotheses = (
        {} if daughter_mass_hypotheses is None else daughter_mass_hypotheses
    )
    unknown_hypotheses = set(mass_hypotheses) - set(daughter_base_names)
    if unknown_hypotheses:
        raise ValueError(
            "Mass hypotheses were supplied for daughters that are not being "
            f"reconstructed: {sorted(unknown_hypotheses)}"
        )

    required = {*PV_COLUMNS, *SIGNAL_B_POSITION_COLUMNS}
    for base_name in daughter_base_names:
        required.update((f"{base_name}_PX", f"{base_name}_PY", f"{base_name}_PZ"))
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")

    result_df = df if inplace else df.copy()
    number_of_rows = len(result_df)
    momentum = np.zeros((number_of_rows, 3), dtype=np.float64)
    energy = np.zeros(number_of_rows, dtype=np.float64)

    for base_name in daughter_base_names:
        daughter_momentum = result_df.loc[
            :, (f"{base_name}_PX", f"{base_name}_PY", f"{base_name}_PZ")
        ].to_numpy(dtype=np.float64, copy=False)
        daughter_momentum_squared = np.sum(
            daughter_momentum * daughter_momentum, axis=1
        )
        momentum += daughter_momentum

        if base_name in mass_hypotheses:
            mass = float(mass_hypotheses[base_name])
            if not np.isfinite(mass) or mass < 0.0:
                raise ValueError(
                    f"Mass hypothesis for {base_name!r} must be finite and non-negative"
                )
            daughter_energy = np.sqrt(daughter_momentum_squared + mass * mass)
        elif f"{base_name}_ENERGY" in result_df.columns:
            daughter_energy = result_df[f"{base_name}_ENERGY"].to_numpy(
                dtype=np.float64, copy=False
            )
        elif f"{base_name}_PE" in result_df.columns:
            daughter_energy = result_df[f"{base_name}_PE"].to_numpy(
                dtype=np.float64, copy=False
            )
        elif f"{base_name}_M" in result_df.columns:
            mass = result_df[f"{base_name}_M"].to_numpy(dtype=np.float64, copy=False)
            daughter_energy = np.sqrt(daughter_momentum_squared + mass * mass)
        else:
            raise ValueError(
                f"Cannot determine the energy of daughter {base_name!r}. Supply "
                f"daughter_mass_hypotheses[{base_name!r}] or add one of "
                f"{base_name}_ENERGY, {base_name}_PE, or {base_name}_M."
            )
        energy += daughter_energy

    # Reconstruct the visible mass from the sum of the supplied daughter
    # four-vectors. A substantially negative m^2 is invalid; only tiny
    # floating-point negative values are rounded to zero.
    momentum_squared = np.sum(momentum * momentum, axis=1)
    mass_squared = energy * energy - momentum_squared
    roundoff_tolerance = 1.0e-12 * np.maximum(energy * energy, momentum_squared)
    valid_mass = (
        np.isfinite(mass_squared)
        & np.isfinite(roundoff_tolerance)
        & (mass_squared >= -roundoff_tolerance)
    )
    valid_mass &= np.all(np.isfinite(momentum), axis=1) & np.isfinite(energy)
    invariant_mass = np.full(number_of_rows, np.nan, dtype=np.float64)
    invariant_mass[valid_mass] = np.sqrt(np.maximum(mass_squared[valid_mass], 0.0))

    production_vertex = result_df.loc[:, PV_COLUMNS].to_numpy(
        dtype=np.float64, copy=False
    )
    decay_vertex = result_df.loc[:, SIGNAL_B_POSITION_COLUMNS].to_numpy(
        dtype=np.float64, copy=False
    )
    flight_direction = _directions(
        np.ascontiguousarray(decay_vertex - production_vertex)
    )
    valid_flight = valid_mass & np.all(np.isfinite(flight_direction), axis=1)

    # The transverse component is the part of the B momentum perpendicular to
    # its reconstructed PV-to-SV flight direction.
    longitudinal_momentum = np.sum(momentum * flight_direction, axis=1)
    transverse_momentum = (
        momentum - longitudinal_momentum[:, np.newaxis] * flight_direction
    )
    missing_pt = np.full(number_of_rows, np.nan, dtype=np.float64)
    missing_pt[valid_flight] = np.sqrt(
        np.sum(transverse_momentum[valid_flight] ** 2, axis=1)
    )

    corrected_mass = np.full(number_of_rows, np.nan, dtype=np.float64)
    corrected_mass[valid_flight] = (
        np.sqrt(invariant_mass[valid_flight] ** 2 + missing_pt[valid_flight] ** 2)
        + missing_pt[valid_flight]
    )

    result_df[f"{output_prefix}_energy"] = energy
    result_df[f"{output_prefix}_PX"] = momentum[:, 0]
    result_df[f"{output_prefix}_PY"] = momentum[:, 1]
    result_df[f"{output_prefix}_PZ"] = momentum[:, 2]
    result_df[f"{output_prefix}_invariant_mass"] = invariant_mass
    result_df[f"{output_prefix}_missing_PT"] = missing_pt
    result_df[f"{output_prefix}_corrected_mass"] = corrected_mass
    return result_df


def _evaluate_track_values(
    function: TrackValueFunction, df: pd.DataFrame, function_name: str
) -> np.ndarray:
    """Evaluate and validate a user callback as one float64 value per row."""
    values = np.asarray(function(df), dtype=np.float64)
    # Scalar callbacks are convenient for hypotheses that apply to all tracks.
    if values.ndim == 0:
        values = np.full(len(df), values.item(), dtype=np.float64)
    if values.shape != (len(df),):
        raise ValueError(f"{function_name} must return a scalar or one value per row")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{function_name} returned non-finite values")
    return values


def _stage_masses(
    added_track: np.ndarray,
    previous_energy: np.ndarray,
    track_energy: np.ndarray,
    momentum: np.ndarray,
    poca: np.ndarray,
    production_vertex: np.ndarray,
    uncombined_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate visible and flight-direction-corrected masses for one stage."""
    valid_combination = added_track >= 0

    # Four-vector energies add linearly. previous_energy already contains all
    # members from the preceding stage; added_track supplies the new member.
    energy = np.full(len(added_track), np.nan, dtype=np.float64)
    energy[valid_combination] = (
        previous_energy[valid_combination]
        + track_energy[added_track[valid_combination]]
    )

    # m_vis^2 = E_sum^2 - |p_sum|^2. Clip tiny negative round-off to zero.
    # Rows without an accepted combination retain the configurable default.
    invariant_mass = uncombined_mass.copy()
    mass_squared = energy * energy - np.sum(momentum * momentum, axis=1)
    invariant_mass[valid_combination] = np.sqrt(
        np.maximum(mass_squared[valid_combination], 0.0)
    )

    missing_pt = np.zeros(len(added_track), dtype=np.float64)
    corrected_mass = uncombined_mass.copy()

    # The decay vertex is the POCA reconstructed at this stage, not B_ENDV.
    # The production-to-decay vector therefore defines the reconstructed
    # secondary candidate's flight direction independently at every stage.
    flight_direction = _directions(np.ascontiguousarray(poca - production_vertex))
    valid_flight = valid_combination & np.all(np.isfinite(flight_direction), axis=1)
    # Remove the projection along the flight direction. The magnitude of what
    # remains is |P'_T,missing| used by the corrected-mass definition.
    longitudinal_momentum = np.sum(momentum * flight_direction, axis=1)
    transverse_momentum = (
        momentum - longitudinal_momentum[:, np.newaxis] * flight_direction
    )
    missing_pt[valid_flight] = np.sqrt(
        np.sum(transverse_momentum[valid_flight] ** 2, axis=1)
    )
    corrected_mass[valid_flight] = (
        np.sqrt(invariant_mass[valid_flight] ** 2 + missing_pt[valid_flight] ** 2)
        + missing_pt[valid_flight]
    )
    corrected_mass[valid_combination & ~valid_flight] = np.nan
    return energy, invariant_mass, missing_pt, corrected_mass


def _candidate_layout(candidate_entries: pd.Series) -> tuple[np.ndarray, ...]:
    """Group equal candidate entries and provide each sorted row's group bounds.

    Returns the permutation into candidate-contiguous order plus two arrays
    mapping every sorted row to the first and one-past-last row of its group.
    The reconstruction kernel can then scan only tracks from that candidate.
    """
    # Integer factorization is substantially cheaper to sort than the string
    # candidate_entry values and preserves their exact equality semantics.
    candidate_codes, _ = pd.factorize(candidate_entries, sort=False)
    order = np.argsort(candidate_codes, kind="stable")
    sorted_codes = candidate_codes[order]
    n_tracks = len(order)
    # Locate transitions between candidates in the sorted representation.
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1,
            np.array([n_tracks], dtype=np.int64),
        )
    )
    # Repeat each boundary according to its group size so that bounds can be
    # indexed directly by row inside Numba, without a binary search.
    sizes = np.diff(starts)
    candidate_first = np.repeat(starts[:-1], sizes)
    candidate_last = np.repeat(starts[1:], sizes)
    return order, candidate_first, candidate_last


def _add_stage_columns(
    df: pd.DataFrame,
    body_count: int,
    order: np.ndarray,
    sorted_added_track: np.ndarray,
    sorted_doca: np.ndarray,
    sorted_poca: np.ndarray,
    sorted_momentum: np.ndarray,
    sorted_vertex_distance: np.ndarray,
    sorted_invariant_mass: np.ndarray,
    sorted_missing_pt: np.ndarray,
    sorted_corrected_mass: np.ndarray,
) -> None:
    """Scatter one stage from candidate-sorted arrays into dataframe row order."""
    n_tracks = len(df)

    # The kernel stores sorted-array positions. Map both the result rows and
    # their added-track references back to original dataframe row positions.
    added_track = np.full(n_tracks, -1, dtype=np.int64)
    valid = sorted_added_track >= 0
    added_track[order[valid]] = order[sorted_added_track[valid]]

    doca = np.full(n_tracks, np.nan, dtype=np.float64)
    poca = np.full((n_tracks, 3), np.nan, dtype=np.float64)
    momentum = np.full((n_tracks, 3), np.nan, dtype=np.float64)
    vertex_distance = np.full(n_tracks, np.nan, dtype=np.float64)
    invariant_mass = np.empty(n_tracks, dtype=np.float64)
    missing_pt = np.empty(n_tracks, dtype=np.float64)
    corrected_mass = np.empty(n_tracks, dtype=np.float64)
    doca[order] = sorted_doca
    poca[order] = sorted_poca
    momentum[order] = sorted_momentum
    vertex_distance[order] = sorted_vertex_distance
    invariant_mass[order] = sorted_invariant_mass
    missing_pt[order] = sorted_missing_pt
    corrected_mass[order] = sorted_corrected_mass

    # A numeric prefix keeps the schema predictable for arbitrary body counts.
    prefix = f"{body_count}_body"
    df[f"{prefix}_added_track_index"] = added_track
    df[f"{prefix}_DOCA"] = doca
    df[f"{prefix}_POCA_X"] = poca[:, 0]
    df[f"{prefix}_POCA_Y"] = poca[:, 1]
    df[f"{prefix}_POCA_Z"] = poca[:, 2]
    df[f"{prefix}_PX"] = momentum[:, 0]
    df[f"{prefix}_PY"] = momentum[:, 1]
    df[f"{prefix}_PZ"] = momentum[:, 2]
    df[f"{prefix}_invariant_mass"] = invariant_mass
    df[f"{prefix}_missing_PT"] = missing_pt
    df[f"{prefix}_corrected_mass"] = corrected_mass
    if body_count >= 3:
        df[f"{prefix}_vertex_distance"] = vertex_distance


def add_n_body_candidate_features(
    df: pd.DataFrame,
    maximum_number_of_tracks: int,
    *,
    maximum_doca: float = 0.15,
    upstream_tolerance: float = 0.1,
    mass_hypothesis: TrackValueFunction = kaon_mass_hypothesis,
    uncombined_mass: TrackValueFunction = zero_uncombined_mass,
    inplace: bool = True,
    show_progress: bool = True,
    row_batch_size: int = 200_000,
) -> pd.DataFrame:
    """Progressively add tracks to reconstruct up to an n-body candidate.

    Every dataframe row starts as a one-track candidate. At each stage, its
    current POCA and summed momentum define a composite track. The unused
    single track from the same ``candidate_entry`` with the smallest DOCA to
    that composite is added. This greedy process is repeated through
    ``maximum_number_of_tracks``.

    For stages of three tracks and above, ``{n}_body_vertex_distance`` is the
    distance between the new POCA and the preceding POCA. It can be used to
    distinguish a prompt common vertex from a spatially separated decay chain;
    no experiment-dependent threshold is imposed here.

    The reconstructed candidate's flight direction at stage n is its current
    ``{n}_body_POCA - B_OWNPV``. Thus, for a cascade reconstructed as a 2-body
    composite plus a single track, the 3-body POCA is the end vertex used by
    the correction. ``{n}_body_missing_PT`` is the magnitude of the combined
    momentum transverse to that direction, and the corrected mass is
    ``sqrt(invariant_mass**2 + missing_PT**2) + missing_PT``.

    ``{n}_body_added_track_index`` contains the zero-based row position of the
    track added at that stage, independent of the dataframe's index labels.
    A stage is accepted only when the DOCA between the preceding composite and
    the new single track is strictly below ``maximum_doca``. Rows that cannot
    reach a stage contain ``-1`` and ``NaN`` geometry outputs, and receive the
    value returned by ``uncombined_mass`` for both mass outputs.

    Parameters
    ----------
    df:
        Track-level dataframe containing ``candidate_entry``, track position,
        and track momentum columns.
    maximum_number_of_tracks:
        Largest combination to construct. Must be at least two.
    maximum_doca:
        Exclusive DOCA limit in mm for each composite-plus-track step.
    upstream_tolerance:
        Maximum downstream displacement in mm allowed for a new POCA relative
        to the preceding fitted vertex, projected along the preceding
        composite momentum. The default permits a 0.1 mm downstream shift to
        accommodate vertex resolution. This constraint starts at the 3-body
        stage because the 2-body POCA is the first fitted vertex.
    mass_hypothesis:
        Vectorized callable receiving the track dataframe and returning either
        one mass per row or a scalar. The default assigns the charged-kaon mass
        to every track. Mass and momentum must use consistent units.
    uncombined_mass:
        Vectorized callable providing the mass value for rows that cannot be
        combined at a stage. The default is zero.
    inplace:
        Mutate ``df`` when true. When false, return an augmented copy.
    show_progress:
        Show one progress bar for each reconstruction stage.
    row_batch_size:
        Number of row-level combinations per parallel Numba call.
    """
    required = {
        CANDIDATE_COLUMN,
        *POSITION_COLUMNS,
        *MOMENTUM_COLUMNS,
        *PV_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")
    if df[CANDIDATE_COLUMN].isna().any():
        raise ValueError("candidate_entry must not contain missing values")
    if not isinstance(maximum_number_of_tracks, int) or maximum_number_of_tracks < 2:
        raise ValueError("maximum_number_of_tracks must be an integer of at least 2")
    if not np.isfinite(maximum_doca) or maximum_doca <= 0.0:
        raise ValueError("maximum_doca must be a positive finite value")
    if not np.isfinite(upstream_tolerance) or upstream_tolerance < 0.0:
        raise ValueError("upstream_tolerance must be a non-negative finite value")
    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")

    # Work on the caller's object unless copy semantics were requested.
    result_df = df if inplace else df.copy()
    n_tracks = len(result_df)
    if n_tracks == 0:
        return result_df

    # User callbacks operate on the unsorted dataframe so they can use any
    # original track columns and naturally return values in dataframe order.
    track_mass = _evaluate_track_values(mass_hypothesis, result_df, "mass_hypothesis")
    if np.any(track_mass < 0.0):
        raise ValueError("mass_hypothesis returned a negative mass")
    result_df[MASS_HYPOTHESIS_COLUMN] = track_mass
    default_mass = _evaluate_track_values(uncombined_mass, result_df, "uncombined_mass")

    # The JIT kernel needs each candidate's tracks in one contiguous slice.
    # Only temporary NumPy arrays are reordered; dataframe order is untouched.
    order, candidate_first, candidate_last = _candidate_layout(
        result_df[CANDIDATE_COLUMN]
    )
    track_positions = np.ascontiguousarray(
        result_df.loc[:, POSITION_COLUMNS].to_numpy(dtype=np.float64, copy=False)[order]
    )
    track_momenta = np.ascontiguousarray(
        result_df.loc[:, MOMENTUM_COLUMNS].to_numpy(dtype=np.float64, copy=False)[order]
    )
    track_directions = _directions(track_momenta)

    # Construct the initial one-track four-vectors under the chosen mass
    # hypothesis. Later stages add one single-track energy and momentum.
    sorted_track_mass = track_mass[order]
    track_energy = np.sqrt(
        np.sum(track_momenta * track_momenta, axis=1) + sorted_track_mass**2
    )
    # B_OWNPV is used as the production vertex of the reconstructed secondary
    # candidate. Every stage supplies its own reconstructed POCA as decay vertex.
    pv = result_df.loc[:, PV_COLUMNS].to_numpy(dtype=np.float64, copy=False)[order]
    production_vertex = np.ascontiguousarray(pv)
    sorted_default_mass = default_mass[order]

    # Each row owns one greedy reconstruction path. members records the sorted
    # row positions already used by that path, preventing track reuse.
    members = np.full((n_tracks, maximum_number_of_tracks), -1, dtype=np.int64)
    members[:, 0] = np.arange(n_tracks)
    previous_poca = track_positions
    previous_momentum = track_momenta
    previous_direction = track_directions
    previous_energy = track_energy

    # Extend 1 -> 2 -> ... -> maximum_number_of_tracks. Outputs from one stage
    # become the composite position, momentum, direction, and energy of the next.
    for body_count in range(2, maximum_number_of_tracks + 1):
        added_track = np.full(n_tracks, -1, dtype=np.int64)
        doca = np.full(n_tracks, np.nan, dtype=np.float64)
        poca = np.full((n_tracks, 3), np.nan, dtype=np.float64)
        momentum = np.full((n_tracks, 3), np.nan, dtype=np.float64)
        vertex_distance = np.full(n_tracks, np.nan, dtype=np.float64)

        for first_row, last_row in _batches(
            n_tracks, row_batch_size, show_progress, body_count
        ):
            _extend_combinations(
                track_positions,
                track_momenta,
                track_directions,
                candidate_first,
                candidate_last,
                previous_poca,
                previous_momentum,
                previous_direction,
                members,
                body_count - 1,
                maximum_doca,
                upstream_tolerance,
                first_row,
                last_row,
                added_track,
                doca,
                poca,
                momentum,
                vertex_distance,
            )

        # Commit the newly selected member before proceeding to the next stage.
        valid = added_track >= 0
        members[valid, body_count - 1] = added_track[valid]
        energy, invariant_mass, missing_pt, corrected_mass = _stage_masses(
            added_track,
            previous_energy,
            track_energy,
            momentum,
            poca,
            production_vertex,
            sorted_default_mass,
        )
        _add_stage_columns(
            result_df,
            body_count,
            order,
            added_track,
            doca,
            poca,
            momentum,
            vertex_distance,
            invariant_mass,
            missing_pt,
            corrected_mass,
        )
        # Failed rows contain NaNs and therefore cannot resume at later stages.
        previous_poca = poca
        previous_momentum = momentum
        previous_direction = _directions(momentum)
        previous_energy = energy

    return result_df


def add_two_body_candidate_features(
    df: pd.DataFrame,
    *,
    maximum_doca: float = 0.15,
    mass_hypothesis: TrackValueFunction = kaon_mass_hypothesis,
    uncombined_mass: TrackValueFunction = zero_uncombined_mass,
    inplace: bool = True,
    show_progress: bool = True,
    row_batch_size: int = 200_000,
) -> pd.DataFrame:
    """Convenience wrapper for two-body-only reconstruction."""
    return add_n_body_candidate_features(
        df,
        maximum_number_of_tracks=2,
        maximum_doca=maximum_doca,
        mass_hypothesis=mass_hypothesis,
        uncombined_mass=uncombined_mass,
        inplace=inplace,
        show_progress=show_progress,
        row_batch_size=row_batch_size,
    )
