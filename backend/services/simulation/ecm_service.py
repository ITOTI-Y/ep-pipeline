from loguru import logger

from backend.models import SimulationJob, SimulationResult
from backend.services.configuration import (
    ECMApply,
    OutputApply,
    PeriodApply,
    SettingApply,
)
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IResultParser,
)
from backend.services.simulation._share import IFileCleaner, ISimulationService
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
        super().__init__(executor, result_parser, file_cleaner, config, job)
        self._ecm_apply = ECMApply()
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)
        self._setting_apply = SettingApply(config=config)

    def prepare(self) -> None:
        logger.info("ECM preparation started")
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._ecm_apply.apply(self._job)
        self._setting_apply.apply(self._job)
        logger.info("ECM preparation completed")

    def execute(self) -> SimulationResult:
        logger.info(f"ECM simulation for job {self._job.id} started")

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
            logger.exception(f"Failed to execute ecm simulation for job {self._job.id}")
            result.add_error(str(e))
            return result

    def cleanup(self) -> None:
        self._file_cleaner.clean(
            job=self._job,
            config=self._config,
        )

    def run(self) -> SimulationResult:
        result = super().run()
        result.ecm_parameters = self._job.ecm_parameters or None
        return result
