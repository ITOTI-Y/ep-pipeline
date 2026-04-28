from typing import NamedTuple, overload

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from backend.citys.core._share import EARTH_RADIUS_KM


class QCResult(NamedTuple):
    final_indices: list[int]
    selection_types: dict[int, str]


@overload
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> np.float64: ...
@overload
def haversine_km(
    lat1: float | NDArray[np.floating],
    lon1: float | NDArray[np.floating],
    lat2: float | NDArray[np.floating],
    lon2: float | NDArray[np.floating],
) -> NDArray[np.float64]: ...
def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance (km) with numpy broadcasting support.

    With ``(M, 1)`` source and ``(1, N)`` target coordinates, returns the
    ``(M, N)`` distance matrix; scalar inputs return a scalar ``np.float64``.
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _check_extreme(indices: list[int], df: pd.DataFrame) -> list[int]:
    current = list(indices)
    checks = [
        ("coldest", "hdd18", True),
        ("hottest", "cdd18", True),
        ("wettest", "annual_mean_dew_point", True),
        ("driest", "annual_mean_dew_point", False),
        ("highest", "elevation", True),
    ]
    for _label, col, use_max in checks:
        idx = int(df[col].idxmax() if use_max else df[col].idxmin())
        if idx not in current:
            current.append(idx)
    return current


def _check_province(
    indices: list[int],
    labels: NDArray,
    x: NDArray,
    meta_df: pd.DataFrame,
) -> list[int]:
    current = list(indices)
    all_provinces = set(meta_df["province"].unique())
    covered = {meta_df.loc[i, "province"] for i in current}
    missing = all_provinces - covered

    for prov in sorted(missing):
        prov_mask = meta_df["province"] == prov
        prov_indices = meta_df.index[prov_mask].tolist()
        best_idx, best_dist = -1, float("inf")
        for idx in prov_indices:
            cluster_label = labels[idx]
            cluster_members = x[labels == cluster_label]
            centroid = cluster_members.mean(axis=0)
            dist = float(np.linalg.norm(x[idx] - centroid))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            current.append(best_idx)

    return current


def _check_forced(
    indices: list[int],
    meta_df: pd.DataFrame,
    forced_names: list[str],
) -> list[int]:
    current = list(indices)
    for name in forced_names:
        name_lower = name.lower()
        matched = [
            i
            for i in meta_df.index
            if meta_df.loc[i, "city"].lower().startswith(name_lower)
        ]
        if not matched:
            continue
        idx = matched[0]
        if idx not in current:
            current.append(idx)
    return current


def run_qc(
    medoid_indices: NDArray[np.intp],
    labels: NDArray[np.intp],
    x: NDArray,
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    forced_cities: list[str],
) -> QCResult:
    selection_types: dict[int, str] = {}

    current = list(medoid_indices)
    for idx in current:
        selection_types[idx] = "medoid"

    extremes = _check_extreme(current, df)
    for idx in extremes:
        if idx not in selection_types:
            selection_types[idx] = "extreme_supplement"

    provinces = _check_province(extremes, labels, x, meta_df)
    for idx in provinces:
        if idx not in selection_types:
            selection_types[idx] = "province_supplement"

    final = _check_forced(provinces, meta_df, forced_cities)
    for idx in final:
        if idx not in selection_types:
            selection_types[idx] = "forced"

    logger.info(f"QC complete: {len(final)} representative cities")
    return QCResult(final, selection_types)
