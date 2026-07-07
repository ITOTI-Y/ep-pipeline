import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
from loguru import logger

from backend.models import SimulationJob
from backend.services.simulation._share import IDataExtractor

_META_QUERY = """
    SELECT ReportDataDictionaryIndex, IsMeter, Name, KeyValue,
           ReportingFrequency, Units
    FROM ReportDataDictionary
"""

_DATA_QUERY = """
    SELECT rd.ReportDataDictionaryIndex, t.Month, t.Day, t.Hour, rd.Value
    FROM ReportData rd
    JOIN Time t ON rd.TimeIndex = t.TimeIndex
    WHERE t.EnvironmentPeriodIndex IN (
        SELECT EnvironmentPeriodIndex FROM EnvironmentPeriods
        WHERE EnvironmentType = 3
    )
    ORDER BY rd.ReportDataDictionaryIndex, rd.TimeIndex
"""

_COLUMNS = [
    "Name",
    "KeyValue",
    "Units",
    "ReportingFrequency",
    "IsMeter",
    "Month",
    "Day",
    "Hour",
    "Value",
]


class DataExtractor(IDataExtractor):
    def extract(self, job: SimulationJob) -> Path:
        output_prefix = job.output_prefix or "eplus"
        sql_path = job.output_directory / f"{output_prefix}out.sql"
        if not sql_path.exists():
            raise FileNotFoundError(f"EnergyPlus SQL output not found: {sql_path}")

        parquet_path = sql_path.with_suffix(".parquet")
        df = self.extract_to_parquet(sql_path, parquet_path)
        logger.success(
            f"Extracted {len(df)} rows "
            f"({df['Name'].nunique()} variables/meters) to {parquet_path}"
        )
        return parquet_path

    def extract_to_parquet(self, sql_path: Path, parquet_path: Path) -> pd.DataFrame:
        with closing(sqlite3.connect(sql_path)) as conn:
            meta = pd.read_sql_query(_META_QUERY, conn)
            df = pd.read_sql_query(_DATA_QUERY, conn)

        for col in ("Month", "Day", "Hour"):
            df[col] = df[col].astype("uint8")
        meta["IsMeter"] = meta["IsMeter"].astype("bool")

        for col in ("Name", "KeyValue", "ReportingFrequency", "Units"):
            meta[col] = meta[col].astype("category")

        df = df.merge(meta, on="ReportDataDictionaryIndex", how="left")
        df = df[_COLUMNS]
        df.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=False)
        return df
