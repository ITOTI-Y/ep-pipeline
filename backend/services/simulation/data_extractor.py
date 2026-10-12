from pathlib import Path
from typing import Final, override

import connectorx as cx
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
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

_CATEGORY_COLUMNS: Final = ("Name", "KeyValue", "Units", "ReportingFrequency")

_COLUMNS: Final = (*_CATEGORY_COLUMNS, "IsMeter", "Month", "Day", "Hour", "Value")

_BATCH_ROWS: Final = 1_000_000


def _read_metadata(uri: str) -> pl.DataFrame:
    meta = pl.from_arrow(cx.read_sql(uri, _META_QUERY, return_type="arrow"))
    if not isinstance(meta, pl.DataFrame):
        raise TypeError(f"Expected a DataFrame for {_META_QUERY!r}")
    return meta.with_columns(
        pl.col("IsMeter").cast(pl.Boolean),
        pl.col(*_CATEGORY_COLUMNS).cast(pl.Categorical),
    )


def _attach_metadata(data: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    joined = data.with_columns(pl.col("Month", "Day", "Hour").cast(pl.UInt8)).join(
        meta, on="ReportDataDictionaryIndex", how="left", maintain_order="left"
    )
    if joined["IsMeter"].null_count():
        raise ValueError("ReportData references an unknown dictionary index")
    return joined.select(_COLUMNS)


def _as_frame(batch: pa.RecordBatch | pa.Table) -> pl.DataFrame:
    frame = pl.from_arrow(batch)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("Expected a DataFrame for a ReportData batch")
    return frame


class DataExtractor(IDataExtractor):
    @override
    def extract(self, job: SimulationJob) -> Path:
        output_prefix = job.output_prefix or "eplus"
        sql_path = job.output_directory / f"{output_prefix}out.sql"
        if not sql_path.exists():
            raise FileNotFoundError(f"EnergyPlus SQL output not found: {sql_path}")

        parquet_path = sql_path.with_suffix(".parquet")
        n_rows, n_variables = self.extract_to_parquet(sql_path, parquet_path)
        logger.success(
            f"Extracted {n_rows} rows "
            f"({n_variables} variables/meters) to {parquet_path}"
        )
        return parquet_path

    def extract_to_parquet(self, sql_path: Path, parquet_path: Path) -> tuple[int, int]:
        uri = f"sqlite://{sql_path.resolve()}"
        meta = _read_metadata(uri)
        reader = cx.read_sql(
            uri, _DATA_QUERY, return_type="arrow_stream", batch_size=_BATCH_ROWS
        )
        empty = _as_frame(reader.schema.empty_table())
        schema = _attach_metadata(empty, meta).to_arrow().schema

        partial_path = parquet_path.with_name(f"{parquet_path.name}.partial")
        n_rows = 0
        names: set[str] = set()
        with pq.ParquetWriter(partial_path, schema, compression="zstd") as writer:
            for batch in reader:
                rows = _attach_metadata(_as_frame(batch), meta)
                writer.write_table(rows.to_arrow())
                n_rows += rows.height
                names.update(rows["Name"].unique().cast(pl.String).to_list())
        partial_path.replace(parquet_path)
        return n_rows, len(names)
