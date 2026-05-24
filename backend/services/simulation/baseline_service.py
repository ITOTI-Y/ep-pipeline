from loguru import logger

from backend.models import SimulationJob, SimulationResult
from backend.services.configuration import OutputApply, PeriodApply, SettingApply
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IResultParser,
)
from backend.services.simulation._share import IFileCleaner, ISimulationService
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
        super().__init__(executor, result_parser, file_cleaner, config, job)
        self._setting_apply = SettingApply(config=config)
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)

    def prepare(self) -> None:
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._setting_apply.apply(self._job)
        logger.info("Baseline preparation completed successfully")

    def execute(self) -> SimulationResult:
        logger.info(f"Executing baseline simulation for job {self._job.id}")

        result = SimulationResult(
            job_id=self._job.id,
            building_type=self._job.idf_file.building_type,
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
            job=self._job, config=self._config, exclude_files=("*.sql", "*.csv")
        )
