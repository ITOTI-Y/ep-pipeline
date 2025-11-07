from eppy.modeleditor import IDF

from backend.domain.models import SimulationContext
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class OutputApply(IApply):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self._config = config

    def apply(self, context: SimulationContext) -> None:
        self._logger.info("Applying output configuration")
        self._configure_output_control_file(context.idf)
        self._configure_output_meter(context.idf)
        self._configure_output_variables(context.idf)
        self._configure_output_controls(context.idf)
        self._logger.info("Output configuration applied successfully")

    def _configure_output_control_file(self, idf: IDF) -> None:
        self._remove_objects(idf, "OUTPUTCONTROL:FILES")

        idf.newidfobject(
            "OUTPUTCONTROL:FILES",
            Output_CSV="Yes",
            Output_MTR="Yes",
            Output_Tabular="Yes",
            Output_SQLite="Yes",
        )

        self._logger.success("Output control file configured successfully")

    def _configure_output_meter(self, idf: IDF) -> None:
        self._remove_objects(idf, "OUTPUT:METER")

        idf.newidfobject(
            "OUTPUT:METER",
            Key_Name="Electricity:Facility",
            Reporting_Frequency="Hourly",
        )

        self._logger.success("Output meter configured successfully")

    def _configure_output_variables(self, idf: IDF) -> None:
        self._remove_objects(idf, "OUTPUT:VARIABLE")

        required_variables = [
            ("Site Outdoor Air Drybulb Temperature", "Hourly"),
            ("Zone Mean Air Temperature", "Hourly"),
            ("Facility Total Electric Demand Power", "Hourly"),
            ("Facility Total Natural Gas Demand Rate", "Hourly"),
            ("Surface Outside Face Incident Solar Radiation Rate per Area", "Hourly"),
            ("Facility Total Purchased Electric Energy", "Monthly"),
            ("Facility Total Natural Gas Energy", "Monthly"),
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

        self._logger.success(f"Added {added_count} output variables to IDF")

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

        self._logger.success("Output controls configured successfully")
