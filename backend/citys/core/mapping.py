import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from backend.citys.core.qc import haversine_km


def map_tmyx_to_dest(
    representative_indices: list[int],
    labels: NDArray[np.intp],
    meta_df: pd.DataFrame,
    dest_coords_df: pd.DataFrame,
) -> pd.DataFrame:

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

    rows = []
    for idx in representative_indices:
        tmyx_name = meta_df.loc[idx, "city_name"]
        tmyx_lat = meta_df.loc[idx, "latitude"]
        tmyx_lon = meta_df.loc[idx, "longitude"]
        cluster = labels[idx]

        exact = dest_coords_df[
            dest_coords_df["city_name"].str.lower()
            == tmyx_name.lower().split("-")[0].split(".")[0]
        ]
        if not exact.empty:
            dest_name = exact.iloc[0]["city_name"]
            rows.append(
                {
                    "tmyx_city": tmyx_name,
                    "dest_city": dest_name,
                    "match_type": "exact",
                    "distance_km": 0.0,
                    "cluster": int(cluster),
                }
            )
            continue

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

        rows.append(
            {
                "tmyx_city": tmyx_name,
                "dest_city": dest_row["city_name"],
                "match_type": match_type,
                "distance_km": float(dists[best_i]),
                "cluster": int(cluster),
            }
        )

    result = pd.DataFrame(rows)
    logger.info(
        f"Mapping: {len(result)} pairs -> {result['dest_city'].nunique()} unique DeST cities, "
        f"same-cluster rate={(result['match_type'] != 'nearest_cross_cluster').mean():.1%}"
    )
    return result
