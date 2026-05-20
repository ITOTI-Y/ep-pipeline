import asyncio
import sqlite3
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from backend.citys.core.qc import haversine_km
from backend.citys.io._share import BUILDING_TYPES

_DISTANCE_THRESHOLD = 50


class DestCoords(TypedDict):
    city_name: str
    province: str
    building_type: str
    latitude: float
    longitude: float
    elevation: float
    file_path: Path


class MatchResult(TypedDict):
    tmyx_city: str
    tmyx_province: str
    tmyx_files: list[Path]
    dest_city: str
    dest_province: str
    dest_files: list[Path]
    match_type: Literal["exact", "nearest_same_cluster", "nearest_cross_cluster"]
    distance_km: float
    cluster: int


def map_tmyx_to_dest(
    representative_indices: pd.DataFrame,
    labels: NDArray[np.intp],
    meta_df: pd.DataFrame,
    dest_dir: Path,
    dest_coords_file: Path,
    out_file: Path,
) -> None:

    if dest_coords_file.exists():
        dest_coords_df = pd.read_csv(dest_coords_file)
    else:
        dest_coords_df = asyncio.run(_get_dest_coords(dest_dir))
        dest_coords_df.to_csv(dest_coords_file, index=False)
    dest_lat_arr = dest_coords_df["latitude"].to_numpy()
    dest_lon_arr = dest_coords_df["longitude"].to_numpy()
    tmyx_lat_arr = meta_df["latitude"].to_numpy()
    tmyx_lon_arr = meta_df["longitude"].to_numpy()

    dist_matrix = haversine_km(
        dest_lat_arr[:, None],
        dest_lon_arr[:, None],
        tmyx_lat_arr[None, :],
        tmyx_lon_arr[None, :],
    )
    nearest_tmyx = np.argmin(dist_matrix, axis=1)

    dest_coords_df = dest_coords_df.copy()
    dest_coords_df["cluster"] = labels[nearest_tmyx]

    results = []
    for _, row in tqdm(
        representative_indices.iterrows(),
        total=len(representative_indices),
        desc="Mapping TMYX to DeST",
    ):
        tmyx_name = row["city"]
        tmyx_lat = row["latitude"]
        tmyx_lon = row["longitude"]
        tmyx_province = row["province"]
        tmyx_file_path = row["file_path"]
        cluster = row["cluster_label"]

        exact = dest_coords_df[
            dest_coords_df["city_name"].str.lower()
            == tmyx_name.lower().split("-")[0].split(".")[0]
        ]
        if not exact.empty:
            exact_dists = haversine_km(
                tmyx_lat,
                tmyx_lon,
                exact["latitude"].to_numpy(),
                exact["longitude"].to_numpy(),
            )
            best_i = int(np.argmin(exact_dists))
            distance = float(exact_dists[best_i])
            dest = exact.iloc[best_i]
            if distance < _DISTANCE_THRESHOLD:
                dest_files = dest_coords_df[
                    (dest_coords_df["province"] == dest["province"])
                    & (dest_coords_df["city_name"] == dest["city_name"])
                ]["file_path"].tolist()
                if len(dest_files) != len(BUILDING_TYPES):
                    logger.warning(
                        f"Exact match found but number of DeST files does not match: {tmyx_name} -> "
                        f"{dest['city_name']}, {len(dest_files)} != {len(BUILDING_TYPES)}"
                    )
                results.append(
                    MatchResult(
                        tmyx_city=tmyx_name,
                        tmyx_province=tmyx_province,
                        tmyx_files=[tmyx_file_path],
                        dest_city=dest["city_name"],
                        dest_province=dest["province"],
                        dest_files=dest_files,
                        match_type="exact",
                        distance_km=distance,
                        cluster=int(cluster),
                    )
                )
                continue
            else:
                logger.warning(
                    f"Exact match found but distance too far: {tmyx_name} -> "
                    f"{dest['city_name']}, distance={distance:.2f} km; falling back to nearest"
                )

        same_cluster = dest_coords_df[dest_coords_df["cluster"] == cluster]
        if same_cluster.empty:
            same_cluster = dest_coords_df

        dists = haversine_km(
            tmyx_lat,
            tmyx_lon,
            same_cluster["latitude"].to_numpy(),
            same_cluster["longitude"].to_numpy(),
        )
        best_i = int(np.argmin(dists))
        dest_row = same_cluster.iloc[best_i]
        match_type = (
            "nearest_same_cluster"
            if len(same_cluster) < len(dest_coords_df)
            else "nearest_cross_cluster"
        )

        dest_files = dest_coords_df[
            (dest_coords_df["province"] == dest_row["province"])
            & (dest_coords_df["city_name"] == dest_row["city_name"])
        ]["file_path"].tolist()
        if len(dest_files) != len(BUILDING_TYPES):
            logger.warning(
                f"Nearest match found but number of DeST files does not match: {tmyx_name} -> "
                f"{dest_row['city_name']}, {len(dest_files)} != {len(BUILDING_TYPES)}"
            )
        results.append(
            MatchResult(
                tmyx_city=tmyx_name,
                tmyx_province=tmyx_province,
                tmyx_files=[tmyx_file_path],
                dest_city=dest_row["city_name"],
                dest_province=dest_row["province"],
                dest_files=dest_files,
                match_type=match_type,
                distance_km=float(dists[best_i]),
                cluster=int(cluster),
            )
        )

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    logger.info(
        f"Mapping: {len(df)} pairs -> {df['dest_city'].nunique()} unique DeST cities, "
        f"same-cluster rate={(df['match_type'] != 'nearest_cross_cluster').mean():.1%}"
    )


async def _get_dest_coords(dir_path: Path) -> pd.DataFrame:
    result = []
    files = dir_path.glob("*.sqlite")

    tasks = [asyncio.to_thread(_parse_one_dest, file_path) for file_path in files]

    pbar = tqdm_async(total=len(tasks), desc="Parsing DeST files", unit="file")
    for task in asyncio.as_completed(tasks):
        try:
            data = await task
            result.append(data)
        except Exception:
            logger.exception("Failed to parse DeST file:")
        finally:
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(result)


def _parse_one_dest(file_path: Path) -> DestCoords:
    building_type = file_path.stem.split("_")[1]
    with sqlite3.connect(str(file_path)) as conn:
        city_name = conn.execute("SELECT city_name FROM environment").fetchone()[0]
        province = conn.execute("SELECT province FROM environment").fetchone()[0]
        latitude = conn.execute("SELECT latitude FROM environment").fetchone()[0]
        longitude = conn.execute("SELECT longitude FROM environment").fetchone()[0]
        elevation = conn.execute("SELECT elevation FROM environment").fetchone()[0]
        return {
            "city_name": city_name,
            "province": province,
            "building_type": building_type,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
            "file_path": file_path.resolve(),
        }
