from dataclasses import dataclass
from typing import Literal

from idfpy import IDF
from idfpy.models.outputs import (
    OutputControlFiles,
    OutputControlTableStyle,
    OutputMeter,
    OutputSQLite,
    OutputTableSummaryReports,
    OutputTableSummaryReportsReportsItem,
    OutputVariable,
)
from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


@dataclass
class ReportingFrequency:
    HOURLY: Literal["Hourly"] = "Hourly"
    DAILY: Literal["Daily"] = "Daily"
    MONTHLY: Literal["Monthly"] = "Monthly"
    ANNUAL: Literal["Annual"] = "Annual"


class OutputApply(IApply):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self._config = config

    def apply(self, job: SimulationJob) -> None:
        logger.info("Applying output configuration")
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        self._configure_output_control_file(job.idf)
        self._configure_output_meter(job.idf)
        self._configure_output_variables(job.idf)
        self._configure_output_controls(job.idf)
        logger.info("Output configuration applied successfully")

    def _configure_output_control_file(self, idf: IDF) -> None:
        self._remove_objects(idf, OutputControlFiles)

        idf.add(
            OutputControlFiles(
                output_csv="Yes",
                output_mtr="Yes",
                output_tabular="Yes",
                output_sqlite="Yes",
            )
        )

        logger.success("Output control file configured successfully")

    def _configure_output_meter(self, idf: IDF) -> None:
        self._remove_objects(idf, OutputMeter)

        meters = [
            "Electricity:Facility",
            "ElectricityNet:Facility",
            "Heating:EnergyTransfer",
            "Cooling:EnergyTransfer",
            "Fans:Electricity",
            "InteriorLights:Electricity",
            "InteriorEquipment:Electricity",
        ]

        for meter in meters:
            idf.add(
                OutputMeter(
                    key_name=meter,
                    reporting_frequency="Hourly",
                )
            )

        logger.success(f"Added {len(meters)} output meters to IDF")

    def _configure_output_variables(self, idf: IDF) -> None:
        self._remove_objects(idf, OutputVariable)

        required_variables = [
            ("Site Outdoor Air Drybulb Temperature", ReportingFrequency.HOURLY),
            ("Site Outdoor Air Wetbulb Temperature", ReportingFrequency.HOURLY),
            ("Site Outdoor Air Relative Humidity", ReportingFrequency.HOURLY),
            ("Site Wind Speed", ReportingFrequency.HOURLY),
            ("Site Wind Direction", ReportingFrequency.HOURLY),
            ("Site Direct Solar Radiation Rate per Area", ReportingFrequency.HOURLY),
            ("Site Diffuse Solar Radiation Rate per Area", ReportingFrequency.HOURLY),
            ("Zone Mean Air Temperature", ReportingFrequency.HOURLY),
            ("Zone Mean Air Humidity Ratio", ReportingFrequency.HOURLY),
            ("Zone Mean Radiant Temperature", ReportingFrequency.HOURLY),
            ("Zone People Occupant Count", ReportingFrequency.HOURLY),
            ("Zone Lights Electricity Rate", ReportingFrequency.HOURLY),
            ("Zone Electric Equipment Electricity Rate", ReportingFrequency.HOURLY),
            ("Zone Infiltration Mass Flow Rate", ReportingFrequency.HOURLY),
            ("Surface Inside Face Temperature", ReportingFrequency.HOURLY),
            ("Surface Outside Face Temperature", ReportingFrequency.HOURLY),
            (
                "Surface Inside Face Conduction Heat Transfer Rate per Area",
                ReportingFrequency.HOURLY,
            ),
            (
                "Surface Outside Face Incident Solar Radiation Rate per Area",
                ReportingFrequency.HOURLY,
            ),
            ("Zone Air System Sensible Heating Rate", ReportingFrequency.HOURLY),
            ("Zone Air System Sensible Cooling Rate", ReportingFrequency.HOURLY),
            ("Zone Mechanical Ventilation Mass Flow Rate", ReportingFrequency.HOURLY),
            ("Zone Thermostat Heating Setpoint Temperature", ReportingFrequency.HOURLY),
            ("Zone Thermostat Cooling Setpoint Temperature", ReportingFrequency.HOURLY),
            ("Facility Total Electricity Demand Rate", ReportingFrequency.HOURLY),
            ("Facility Total Purchased Electricity Rate", ReportingFrequency.MONTHLY),
            ("Generator Produced DC Electricity Rate", ReportingFrequency.HOURLY),
            ("Electric Storage Simple Charge State", ReportingFrequency.HOURLY),
            ("Electric Storage Charge Power", ReportingFrequency.HOURLY),
            ("Electric Storage Discharge Power", ReportingFrequency.HOURLY),
        ]

        added_count = 0

        for var_name, frequency in required_variables:
            idf.add(
                OutputVariable(
                    key_value="*",
                    variable_name=var_name,
                    reporting_frequency=frequency,
                )
            )
            added_count += 1

        logger.success(f"Added {added_count} output variables to IDF")

    def _configure_output_controls(self, idf: IDF) -> None:
        self._remove_objects(idf, OutputControlTableStyle)
        self._remove_objects(idf, OutputTableSummaryReports)
        self._remove_objects(idf, OutputSQLite)

        idf.add(
            OutputControlTableStyle(
                column_separator="Comma",
                unit_conversion="JtoKWH",
            )
        )

        idf.add(
            OutputTableSummaryReports(
                reports=[
                    OutputTableSummaryReportsReportsItem(
                        report_name="AllSummaryAndMonthly",
                    ),
                ],
            )
        )

        idf.add(
            OutputSQLite(
                option_type="SimpleAndTabular",
                unit_conversion_for_tabular_data="JtoKWH",
            )
        )

        logger.success("Output controls configured successfully")
