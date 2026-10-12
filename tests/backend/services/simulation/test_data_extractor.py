import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from backend.services.simulation import data_extractor
from backend.services.simulation.data_extractor import DataExtractor

RUN_PERIOD = 3
DESIGN_DAY = 1


def _write_sql(path: Path, report_data: list[tuple[int, int, float]]) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE ReportDataDictionary (
                ReportDataDictionaryIndex INTEGER, IsMeter INTEGER, Name TEXT,
                KeyValue TEXT, ReportingFrequency TEXT, Units TEXT);
            CREATE TABLE EnvironmentPeriods (
                EnvironmentPeriodIndex INTEGER, EnvironmentType INTEGER);
            CREATE TABLE Time (
                TimeIndex INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER,
                EnvironmentPeriodIndex INTEGER);
            CREATE TABLE ReportData (
                ReportDataIndex INTEGER, TimeIndex INTEGER,
                ReportDataDictionaryIndex INTEGER, Value REAL);
            """
        )
        conn.executemany(
            "INSERT INTO ReportDataDictionary VALUES (?, ?, ?, ?, ?, ?)",
            [
                (7, 0, "Zone Air Temperature", "ZONE_1", "Hourly", "C"),
                (9, 1, "Electricity:Facility", "", "Hourly", "J"),
            ],
        )
        conn.executemany(
            "INSERT INTO EnvironmentPeriods VALUES (?, ?)",
            [(1, DESIGN_DAY), (2, RUN_PERIOD)],
        )
        conn.executemany(
            "INSERT INTO Time VALUES (?, ?, ?, ?, ?)",
            [(1, 7, 21, 1, 1), (2, 1, 1, 1, 2), (3, 1, 1, 2, 2)],
        )
        conn.executemany(
            "INSERT INTO ReportData VALUES (?, ?, ?, ?)",
            [(i, *row) for i, row in enumerate(report_data)],
        )
        conn.commit()


def test_run_period_rows_are_streamed_to_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_extractor, "_BATCH_ROWS", 2)
    sql_path, parquet_path = tmp_path / "eplusout.sql", tmp_path / "eplusout.parquet"
    _write_sql(
        sql_path,
        [(1, 7, 99.0), (2, 7, 20.5), (3, 7, 21.0), (2, 9, 1e6), (3, 9, 2e6)],
    )

    n_rows, n_variables = DataExtractor().extract_to_parquet(sql_path, parquet_path)

    df = pd.read_parquet(parquet_path)
    assert (n_rows, n_variables) == (4, 2)
    assert df["Value"].tolist() == [20.5, 21.0, 1e6, 2e6]
    assert (
        df["Name"].astype(str).tolist()
        == ["Zone Air Temperature"] * 2 + ["Electricity:Facility"] * 2
    )
    assert df["IsMeter"].tolist() == [False, False, True, True]
    assert df["Hour"].tolist() == [1, 2, 1, 2]
    assert str(df["KeyValue"].dtype) == "category"
    assert str(df["Month"].dtype) == "uint8"
    assert not list(tmp_path.glob("*.partial"))


def test_unknown_dictionary_index_raises_without_parquet(tmp_path: Path) -> None:
    sql_path, parquet_path = tmp_path / "eplusout.sql", tmp_path / "eplusout.parquet"
    _write_sql(sql_path, [(2, 8, 1.0)])

    with pytest.raises(ValueError, match="unknown dictionary index"):
        DataExtractor().extract_to_parquet(sql_path, parquet_path)

    assert not parquet_path.exists()
