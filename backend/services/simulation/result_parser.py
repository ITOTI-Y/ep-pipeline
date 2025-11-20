import sqlite3
from pathlib import Path

import pandas as pd
from loguru import logger

from backend.models import SimulationJob, SimulationResult, Surface
from backend.services.interfaces import IResultParser


class ResultParser(IResultParser):
    def __init__(self):
        self._logger = logger.bind(module=self.__class__.__name__)

    def parse(
        self,
        result: SimulationResult,
        job: SimulationJob,
    ) -> SimulationResult:
        result.table_csv_path = (
            job.output_directory / f"{job.output_prefix}tbl.csv"
        )
        result.meter_csv_path = (
            job.output_directory / f"{job.output_prefix}mtr.csv"
        )
        result.variables_csv_path = (
            job.output_directory / f"{job.output_prefix}out.csv"
        )
        result.sql_path = (
            job.output_directory / f"{job.output_prefix}out.sql"
        )

        if result.sql_path.exists():
            self._parse_from_sql(result, result.sql_path)
        return result

    def _parse_from_sql(self, result: SimulationResult, sql_path: Path) -> None:
        conn = sqlite3.connect(str(sql_path))
        try:
            self._parse_energy_from_sql(result, conn)
            self._parse_area_from_sql(result, conn)
            self._parse_irradiation_from_sql(result, conn)
        finally:
            conn.close()

    def _parse_energy_from_sql(
        self, result: SimulationResult, conn: sqlite3.Connection
    ) -> None:
        try:
            query = self.ENERGY_QUERY
            df = pd.read_sql_query(query, conn)
            key_mapping = self.ENERGY_KEY_MAPPING
            for _, row in df.iterrows():
                row_name = str(row["RowName"])
                column_name = str(row["ColumnName"])
                if row_name in key_mapping and column_name in key_mapping[row_name]:
                    attr_name = key_mapping[row_name][column_name]
                    setattr(result, attr_name, float(row["Value"]))
        except Exception as e:
            self._logger.exception("Failed to parse energy from SQL: ")
            result.add_error(f"Failed to parse energy from SQL: {e}")

    def _parse_area_from_sql(
        self, result: SimulationResult, conn: sqlite3.Connection
    ) -> None:
        try:
            query = self.AREA_QUERY
            df = pd.read_sql_query(query, conn)
            key_mapping = self.AREA_KEY_MAPPING
            for _, row in df.iterrows():
                row_name = str(row["RowName"])
                if row_name in key_mapping:
                    setattr(result, key_mapping[row_name], float(row["Value"]))
        except Exception as e:
            self._logger.exception("Failed to parse area from SQL: ")
            result.add_error(f"Failed to parse area from SQL: {e}")

    def _parse_irradiation_from_sql(
        self, result: SimulationResult, conn: sqlite3.Connection
    ) -> None:
        try:
            query = self.IRRADIATION_QUERY
            df = pd.read_sql_query(query, conn)
            for _, row in df.iterrows():
                result.surfaces.append(
                    Surface(
                        name=str(row["name"]),
                        type=str(row["type"]),
                        hour_count=int(row["hour_count"]),
                        sum_irradiation=float(
                            row["sum_irradiation"]
                            * self.IRRADIATION_UNIT_TO_HOURS[str(row["frequency"])]
                        )
                        / 1000,
                        unit="kWh/m²"
                        if str(row["unit"]) == "W/m2"
                        else str(row["unit"] + "* h"),
                    )
                )
        except Exception as e:
            self._logger.exception("Failed to parse irradiation from SQL: ")
            result.add_error(f"Failed to parse irradiation from SQL: {e}")
