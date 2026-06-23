from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pandas as pd
from loguru import logger

from backend.citys.io.geo import lookup_province

EPW_COL_DRY_BULB_TEMPERATURE: Final = 6
EPW_COL_DEW_POINT_TEMPERATURE: Final = 7
EPW_COL_GLOBAL_HORIZONTAL_RADIATION: Final = 13
EPW_COL_DIFFUSE_HORIZONTAL_RADIATION: Final = 15
EPW_COL_WIND_SPEED: Final = 21
EPW_COL_MONTH: Final = 1

REMOVE_LIST: Final = [".AP", ".Intl"]


class EPWHeader(TypedDict):
    city: str
    province: str
    wmo_id: str
    latitude: float
    longitude: float
    elevation: float


def _parse_epw_header(path: Path) -> EPWHeader:
    with open(path, encoding="utf-8", errors="replace") as f:
        line1 = f.readline().strip().split(",")
        for remove in REMOVE_LIST:
            line1[1] = line1[1].removesuffix(remove)
    latitude = float(line1[6])
    longitude = float(line1[7])
    province = lookup_province(latitude, longitude)
    return EPWHeader(
        city=line1[1].split("-")[0],
        province=province,
        wmo_id=line1[5],
        latitude=latitude,
        longitude=longitude,
        elevation=float(line1[9]),
    )


def _parse_epw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=8, header=None, encoding="utf-8")


def _extract_one(path: Path) -> dict:
    header = _parse_epw_header(path)
    df = _parse_epw_data(path)
    temp = df[EPW_COL_DRY_BULB_TEMPERATURE].values
    dew = df[EPW_COL_DEW_POINT_TEMPERATURE].values
    ghi = df[EPW_COL_GLOBAL_HORIZONTAL_RADIATION].values
    dhi = df[EPW_COL_DIFFUSE_HORIZONTAL_RADIATION].values
    wind = df[EPW_COL_WIND_SPEED].values
    month = df[EPW_COL_MONTH].values

    row = {}
    row.update(header)

    row["hdd18"] = np.sum(np.maximum(0, 18 - temp)) / 24.0
    row["cdd18"] = np.sum(np.maximum(0, temp - 18)) / 24.0
    row["annual_mean_dew_point"] = float(np.mean(dew))
    row["annual_ghi"] = float(np.sum(ghi))
    row["annual_dhi"] = float(np.sum(dhi))
    row["annual_mean_wind_speed"] = float(np.mean(wind))

    for m in range(1, 13):
        mask = month == m
        row[f"temp_m{m:02d}"] = float(np.mean(temp[mask]))
        row[f"ghi_m{m:02d}"] = float(np.mean(ghi[mask]))

    return row


def extract_all(epw_dir: Path, output_path: Path) -> pd.DataFrame:
    epw_files = sorted(epw_dir.glob("*.epw"))
    if not epw_files:
        raise ValueError(f"No EPW files found in {epw_dir}")

    rows = []
    for f in epw_files:
        try:
            rows.append(_extract_one(f))
        except Exception:
            logger.opt(exception=True).warning(f"Failed to extract features from {f}")
    logger.info(f"Extracted features from {len(rows)}/{len(epw_files)} EPW files")
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df
