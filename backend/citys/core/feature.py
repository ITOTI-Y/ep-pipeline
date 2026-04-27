from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from loguru import logger
from pydantic.dataclasses import dataclass


@dataclass
class EPWFile:
    dry_bulb_temperature: int = 6
    dew_point_temperature: int = 7
    global_horizontal_radiation: int = 13  # Wh/m²
    direct_normal_radiation: int = 15  # Wh/m²
    wind_speed: int = 21  # m/s
    month: int = 1


class EPWHeader(TypedDict):
    city_name: str
    province: str
    wmo_id: str
    latitude: float
    longitude: float
    elevation: float


def _parse_epw_header(path: Path) -> EPWHeader:
    with open(path, encoding="utf-8", errors="replace") as f:
        line1 = f.readline().strip().split(",")
    return EPWHeader(
        city_name=line1[1],
        province=line1[3],
        wmo_id=line1[5],
        latitude=float(line1[6]),
        longitude=float(line1[7]),
        elevation=float(line1[9]),
    )


def _parse_epw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=8, header=None, encoding="utf-8")


def _extract_one(path: Path) -> dict:
    header = _parse_epw_header(path)
    df = _parse_epw_data(path)
    temp = df[EPWFile.dry_bulb_temperature].values
    dew = df[EPWFile.dew_point_temperature].values
    ghi = df[EPWFile.global_horizontal_radiation].values
    dhi = df[EPWFile.direct_normal_radiation].values
    wind = df[EPWFile.wind_speed].values
    month = df[EPWFile.month].values

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
            logger.warning(f"Failed to extract features from {f}")
    logger.info(f"Extracted features from {len(rows)}/{len(epw_files)} EPW files")
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df
