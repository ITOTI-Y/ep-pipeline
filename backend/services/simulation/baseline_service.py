from eppy.modeleditor import IDF
from loguru import logger

from backend.models import SimulationJob, SimulationResult
from backend.services.configuration import OutputApply, PeriodApply
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IFileCleaner,
    IResultParser,
    ISimulationService,
)
from backend.utils.config import ConfigManager


class BaselineService(ISimulationService):
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
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)

    def prepare(self) -> None:
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        logger.info("Baseline preparation completed successfully")

    def execute(self) -> SimulationResult:
        logger.info(f"Executing baseline simulation for job {self._job.id}")

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
            logger.exception("Failed to execute baseline simulation")
            result.add_error(str(e))
            return result

    def cleanup(self) -> None:
        self._file_cleaner.clean(
            job=self._job,
            config=self._config,
            exclude_files=("*.sql","*.csv")
        )

