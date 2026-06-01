from loguru import logger

from backend.models import SimulationJob, SimulationResult
from backend.services.configuration import (
    ECMApply,
    OutputApply,
    PeriodApply,
    SettingApply,
)
from backend.services.interfaces import ISimulationService
from backend.services.simulation.executor import EnergyPlusExecutor
from backend.services.simulation.file_cleaner import FileCleaner
from backend.services.simulation.result_parser import ResultParser
from backend.utils.config import ConfigManager


class ECMService(ISimulationService):
    _cleanup_exclude: tuple[str, ...] = ()

    def __init__(
        self,
        executor: EnergyPlusExecutor,
        result_parser: ResultParser,
        file_cleaner: FileCleaner,
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
        self._setting_apply = SettingApply(config=config)

    def prepare(self) -> None:
        logger.info("ECM preparation started")
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._ecm_apply.apply(self._job)
        self._setting_apply.apply(self._job)
        logger.info("ECM preparation completed")

    def run(self) -> SimulationResult:
        result = super().run()
        result.ecm_parameters = self._job.ecm_parameters or None
        return result
