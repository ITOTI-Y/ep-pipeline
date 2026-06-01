from loguru import logger

from backend.models import SimulationJob, Surface
from backend.services.configuration import (
    OutputApply,
    PeriodApply,
    PVApply,
    ScheduleApply,
    SettingApply,
    StorageApply,
)
from backend.services.interfaces import ISimulationService
from backend.services.simulation.executor import EnergyPlusExecutor
from backend.services.simulation.file_cleaner import FileCleaner
from backend.services.simulation.result_parser import ResultParser
from backend.utils.config import ConfigManager


class PVService(ISimulationService):
    def __init__(
        self,
        executor: EnergyPlusExecutor,
        result_parser: ResultParser,
        file_cleaner: FileCleaner,
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
        self._schedule_apply = ScheduleApply(config=config)
        self._setting_apply = SettingApply(config=config)

    def prepare(self) -> None:
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._pv_apply.apply(self._job)
        self._schedule_apply.apply(self._job)
        self._storage_apply.apply(self._job)
        self._setting_apply.apply(self._job)
        logger.info("PV preparation completed successfully")
