import sqlite3
from pathlib import Path
from typing import ClassVar

import pandas as pd
from loguru import logger

from backend.models import SimulationJob, SimulationResult, Surface


class ResultParser:
    ENERGY_QUERY = """
            SELECT
                RowName,
                ColumnName,
                Value
            FROM
                TabularDataWithStrings
            WHERE
                ReportName = 'AnnualBuildingUtilityPerformanceSummary'
                AND TableName = 'Site and Source Energy'
                AND RowName IN ('Total Site Energy', 'Total Source Energy', 'Net Site Energy', 'Net Source Energy')
                AND ColumnName IN ('Total Energy', 'Energy Per Conditioned Building Area')
                AND Units IN ('kWh', 'kWh/m2')
            """
    AREA_QUERY = """
            SELECT
                RowName,
                ColumnName,
                Value
            FROM
                TabularDataWithStrings
            WHERE
                ReportName = 'AnnualBuildingUtilityPerformanceSummary'
                AND TableName = 'Building Area'
                AND RowName IN ('Total Building Area', 'Net Conditioned Building Area')
                AND Units = 'm2'
            """
    IRRADIATION_QUERY = """
                SELECT
                    r.KeyValue as name,
                    s.ClassName as type,
                    COUNT(*) as hour_count,
                    SUM(r.value) as sum_irradiation,
                    r.ReportingFrequency as frequency,
                    r.Units as unit
                FROM
                    ReportVariableWithTime r
                JOIN
                    Surfaces s on s.SurfaceName = r.KeyValue
                WHERE
                    r.Name = 'Surface Outside Face Incident Solar Radiation Rate per Area'
                    AND s.ClassName IN ('Wall', 'Roof')
                GROUP BY
                    r.KeyValue
            """
    ENERGY_KEY_MAPPING: ClassVar[dict[str, dict[str, str]]] = {
        "Total Site Energy": {
            "Total Energy": "total_site_energy",
            "Energy Per Conditioned Building Area": "total_site_eui",
        },
        "Total Source Energy": {
            "Total Energy": "total_source_energy",
            "Energy Per Conditioned Building Area": "total_source_eui",
        },
        "Net Site Energy": {
            "Total Energy": "net_site_energy",
            "Energy Per Conditioned Building Area": "net_site_eui",
        },
        "Net Source Energy": {
            "Total Energy": "net_source_energy",
            "Energy Per Conditioned Building Area": "net_source_eui",
        },
    }
    AREA_KEY_MAPPING: ClassVar[dict[str, str]] = {
        "Total Building Area": "total_building_area",
        "Net Conditioned Building Area": "net_building_area",
    }
    IRRADIATION_UNIT_TO_HOURS: ClassVar[dict[str, int]] = {
        "Hourly": 1,
        "Daily": 24,
        "Monthly": 30 * 24,
        "Annual": 365 * 24,
    }

    def parse(
        self,
        result: SimulationResult,
        job: SimulationJob,
    ) -> SimulationResult:
        result.table_csv_path = job.output_directory / f"{job.output_prefix}tbl.csv"
        result.meter_csv_path = job.output_directory / f"{job.output_prefix}mtr.csv"
        result.variables_csv_path = job.output_directory / f"{job.output_prefix}out.csv"
        result.sql_path = job.output_directory / f"{job.output_prefix}out.sql"

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
            logger.exception("Failed to parse energy from SQL: ")
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
            logger.exception("Failed to parse area from SQL: ")
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
            logger.exception("Failed to parse irradiation from SQL: ")
            result.add_error(f"Failed to parse irradiation from SQL: {e}")
