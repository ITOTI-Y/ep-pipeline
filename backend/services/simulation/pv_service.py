from loguru import logger

from backend.models import SimulationJob, Surface
from backend.services.configuration import (
    GeneralApply,
    OutputApply,
    PeriodApply,
    PVApply,
    ScheduleApply,
    StorageApply,
)
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IResultParser,
)
from backend.services.simulation._share import IFileCleaner, ISimulationService
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
        super().__init__(executor, result_parser, file_cleaner, config, job)
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)
        self._pv_apply = PVApply(config=config, surfaces=surfaces)
        self._storage_apply = StorageApply(
            config=config, building_type=job.idf_file.building_type
        )
        self._schedule_apply = ScheduleApply(config=config)
        self._general_apply = GeneralApply(config=config)

    def prepare(self) -> None:
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._pv_apply.apply(self._job)
        self._schedule_apply.apply(self._job)
        self._storage_apply.apply(self._job)
        self._general_apply.apply(self._job)
        logger.info("PV preparation completed successfully")

    def cleanup(self) -> None:
        self._file_cleaner.clean(job=self._job, config=self._config)
