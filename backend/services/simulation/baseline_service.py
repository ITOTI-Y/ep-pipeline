from eppy.modeleditor import IDF
from loguru import logger

from backend.domain.models import BaselineContext, SimulationResult
from backend.services.configuration import OutputApply, PeriodApply
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
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)

    def prepare(self, context: BaselineContext) -> None:
        self._output_apply.apply(context)
        self._period_apply.apply(context)
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
