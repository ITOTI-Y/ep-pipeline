from eppy.modeleditor import IDF
from loguru import logger

from backend.domain.models import BaselineContext, SimulationResult
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IFileCleaner,
    IResultParser,
    ISimulationService,
)
from backend.utils.config import ConfigManager


class BaselineService(ISimulationService[BaselineContext]):
    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: IResultParser,
        file_cleaner: IFileCleaner,
        config: ConfigManager,
    ):
        self._executor = executor
        self._result_parser = result_parser
        self._file_cleaner = file_cleaner
        self._config = config
        self._logger = logger.bind(service=self.__class__.__name__)

    def prepare(self, context: BaselineContext) -> None:
        self._configure_output_control_file(context.idf)
        self._configure_output_meter(context.idf)
        self._configure_output_variables(context.idf)
        self._configure_output_controls(context.idf)
        self._configure_simulation_period(context.idf)
        self._logger.info("Preparation completed successfully")

    def execute(self, context: BaselineContext) -> SimulationResult:
        self._logger.info(f"Executing baseline simulation for job {context.job.id}")

        try:
            result = self._executor.run(
                context=context,
            )

            result = self._result_parser.parse(
                result=result,
                context=context,
            )
            return result
        except Exception as e:
            self._logger.exception("Failed to execute baseline simulation")
            result.add_error(str(e))
            return result

    def cleanup(self, context: BaselineContext) -> None:
        self._file_cleaner.clean(
            context=context,
            config=self._config,
        )

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

    def _configure_simulation_period(self, idf: IDF) -> None:
        self._remove_objects(idf, "RUNPERIOD")

        idf.newidfobject(
            "RUNPERIOD",
            Name="Default Run Period",
            Begin_Month=self._config.simulation.begin_month,
            Begin_Day_of_Month=self._config.simulation.begin_day,
            Begin_Year=self._config.simulation.begin_year,
            End_Month=self._config.simulation.end_month,
            End_Day_of_Month=self._config.simulation.end_day,
            End_Year=self._config.simulation.end_year,
        )

        self._logger.success("Simulation period configured successfully")
