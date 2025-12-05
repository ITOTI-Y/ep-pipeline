from loguru import logger

from backend.models import SimulationJob, SimulationResult, Surface
from backend.services.configuration import (
    OutputApply,
    PeriodApply,
    PVApply,
    StorageApply,
)
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IFileCleaner,
    IResultParser,
    ISimulationService,
)
from backend.utils.config import ConfigManager


class PVService(ISimulationService):
    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: IResultParser,
        file_cleaner: IFileCleaner,
        config: ConfigManager,
        job: SimulationJob,
        surfaces: list[Surface],
    ):
        self._config = config
        self._job = job
        self._executor = executor
        self._result_parser = result_parser
        self._file_cleaner = file_cleaner
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)
        self._pv_apply = PVApply(config=config, surfaces=surfaces)
        self._storage_apply = StorageApply(
            config=config, building_type=job.building.building_type
        )

    def prepare(self) -> None:
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._pv_apply.apply(self._job)
        self._storage_apply.apply(self._job)
        logger.info("PV preparation completed successfully")

    def execute(self) -> SimulationResult:
        logger.info(f"Executing PV simulation for job {self._job.id}")

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
            logger.exception("Failed to execute PV simulation")
            result.add_error(str(e))
            return result

    def cleanup(self) -> None:
        self._file_cleaner.clean(
            job=self._job,
            config=self._config,
            exclude_files=("*.sql","*.csv")
        )

    def run(self) -> SimulationResult:
        try:
            self.prepare()
            result = self.execute()
            return result
        finally:
            self.cleanup()
