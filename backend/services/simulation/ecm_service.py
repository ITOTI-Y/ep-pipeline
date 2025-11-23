from loguru import logger

from backend.models import SimulationJob, SimulationResult
from backend.services.configuration import ECMApply, OutputApply, PeriodApply
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IFileCleaner,
    IResultParser,
    ISimulationService,
)
from backend.utils.config import ConfigManager


class ECMService(ISimulationService):
    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: IResultParser,
        file_cleaner: IFileCleaner,
        config: ConfigManager,
        job: SimulationJob,
    ):
        self._job = job
        self._executor = executor
        self._result_parser = result_parser
        self._file_cleaner = file_cleaner
        self._config = config
        self._ecm_apply = ECMApply()
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)

    def prepare(self) -> None:
        logger.info("ECM preparation started")
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._ecm_apply.apply(self._job, self._job.ecm_parameters)  # type: ignore
        logger.info("ECM preparation completed")

    def execute(self) -> SimulationResult:
        logger.info(f"ECM simulation for job {self._job.id} started")

        result = SimulationResult(
            job_id=self._job.id,
            building_type=self._job.building.building_type,
        )

        try:
            result = self._executor.run(
                job=self._job,
            )
            result = self._result_parser.parse(
                result=result,
                job=self._job,
            )
            return result
        except Exception as e:
            logger.exception(f"Failed to execute ecm simulation for job {self._job.id}")
            result.add_error(str(e))
            return result

    def cleanup(self) -> None:
        self._file_cleaner.clean(
            job=self._job,
            config=self._config,
        )

    def run(self) -> SimulationResult:
        try:
            self.prepare()
            result = self.execute()
            result.ecm_parameters = self._job.ecm_parameters or None
            return result
        finally:
            self.cleanup()
