from loguru import logger

from backend.models import ECMContext, ECMParameters, SimulationResult
from backend.services.configuration import ECMApply, OutputApply, PeriodApply
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IFileCleaner,
    IResultParser,
    ISimulationService,
)
from backend.utils.config import ConfigManager


class ECMService(ISimulationService[ECMContext]):
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
        self._ecm_apply = ECMApply()
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)
        self._logger = logger.bind(service=self.__class__.__name__)

    def prepare(self, context: ECMContext, ecm_parameters: ECMParameters) -> None:
        self._logger.info("Preparing ECM simulation")
        self._output_apply.apply(context)
        self._period_apply.apply(context)
        self._ecm_apply.apply(context, ecm_parameters)
        self._logger.info("ECM preparation completed successfully")

    def execute(self, context: ECMContext) -> SimulationResult:
        self._logger.info(f"Executing ecm simulation for job {context.job.id}")
        result = SimulationResult(
            job_id=context.job.id,
            success=False,
        )
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
            self._logger.exception(
                f"Failed to execute ecm simulation for job {context.job.id}"
            )
            result.add_error(str(e))
            return result

    def cleanup(self, context: ECMContext) -> None:
        self._file_cleaner.clean(
            context=context,
            config=self._config,
        )

    def run(
        self, context: ECMContext, ecm_parameters: ECMParameters
    ) -> SimulationResult:
        try:
            self.prepare(context, ecm_parameters)
            result = self.execute(context)
            result.ecm_parameters = ecm_parameters
            return result
        finally:
            self.cleanup(context)
