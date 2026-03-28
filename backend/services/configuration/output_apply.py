from eppy.modeleditor import IDF
from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


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
        self._remove_objects(idf, "OUTPUTCONTROL:FILES")

        idf.newidfobject(
            "OUTPUTCONTROL:FILES",
            Output_CSV="Yes",
            Output_MTR="Yes",
            Output_Tabular="Yes",
            Output_SQLite="Yes",
        )

        logger.success("Output control file configured successfully")

    def _configure_output_meter(self, idf: IDF) -> None:
        self._remove_objects(idf, "OUTPUT:METER")

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
            idf.newidfobject(
                "OUTPUT:METER",
                Key_Name=meter,
                Reporting_Frequency="Hourly",
            )

        logger.success(f"Added {len(meters)} output meters to IDF")

    def _configure_output_variables(self, idf: IDF) -> None:
        self._remove_objects(idf, "OUTPUT:VARIABLE")

        required_variables = [
            ("Site Outdoor Air Drybulb Temperature", "Hourly"),
            ("Site Outdoor Air Wetbulb Temperature", "Hourly"),
            ("Site Outdoor Air Relative Humidity", "Hourly"),
            ("Site Wind Speed", "Hourly"),
            ("Site Wind Direction", "Hourly"),
            ("Site Direct Solar Radiation Rate per Area", "Hourly"),
            ("Site Diffuse Solar Radiation Rate per Area", "Hourly"),
            ("Zone Mean Air Temperature", "Hourly"),
            ("Zone Mean Air Humidity Ratio", "Hourly"),
            ("Zone Mean Radiant Temperature", "Hourly"),
            ("Zone People Occupant Count", "Hourly"),
            ("Zone Lights Electricity Rate", "Hourly"),
            ("Zone Electric Equipment Electricity Rate", "Hourly"),
            ("Zone Infiltration Mass Flow Rate", "Hourly"),
            ("Surface Inside Face Temperature", "Hourly"),
            ("Surface Outside Face Temperature", "Hourly"),
            ("Surface Inside Face Conduction Heat Transfer Rate per Area", "Hourly"),
            ("Surface Outside Face Incident Solar Radiation Rate per Area", "Hourly"),
            ("Zone Air System Sensible Heating Rate", "Hourly"),
            ("Zone Air System Sensible Cooling Rate", "Hourly"),
            ("Zone Mechanical Ventilation Mass Flow Rate", "Hourly"),
            ("Zone Thermostat Heating Setpoint Temperature", "Hourly"),
            ("Zone Thermostat Cooling Setpoint Temperature", "Hourly"),
            ("Facility Total Electricity Demand Rate", "Hourly"),
            ("Facility Total Purchased Electricity Rate", "Monthly"),
            ("Generator Produced DC Electricity Rate", "Hourly"),
            ("Electric Storage Simple Charge State", "Hourly"),
            ("Electric Storage Charge Power", "Hourly"),
            ("Electric Storage Discharge Power", "Hourly"),
        ]

        added_count = 0

        for var_name, frequency in required_variables:
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value="*",
                Variable_Name=var_name,
                Reporting_Frequency=frequency,
            )
            added_count += 1

        logger.success(f"Added {added_count} output variables to IDF")

    def _configure_output_controls(self, idf: IDF) -> None:
        self._remove_objects(idf, "OUTPUTCONTROL:TABLE:STYLE")
        self._remove_objects(idf, "OUTPUT:TABLE:SUMMARYREPORTS")
        self._remove_objects(idf, "OUTPUT:SQLITE")

        idf.newidfobject(
            "OUTPUTCONTROL:TABLE:STYLE",
            Column_Separator="Comma",
            Unit_Conversion="JtoKWH",
        )

        idf.newidfobject(
            "OUTPUT:TABLE:SUMMARYREPORTS",
            Report_1_Name="AllSummaryAndMonthly",
        )

        idf.newidfobject(
            "OUTPUT:SQLITE",
            Option_Type="SimpleAndTabular",
            Unit_Conversion_for_Tabular_Data="JtoKWH",
        )

        logger.success("Output controls configured successfully")
