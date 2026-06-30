from loguru import logger

from backend.models import SimulationJobSchema
from backend.services.configuration import OutputApply, PeriodApply
from backend.services.interfaces import ISimulationService
from backend.services.simulation.executor import EnergyPlusExecutor
from backend.services.simulation.file_cleaner import FileCleaner
from backend.services.simulation.result_parser import ResultParser
from backend.utils.config import ConfigManager


class BaselineService(ISimulationService):
    def __init__(
        self,
        executor: EnergyPlusExecutor,
        result_parser: ResultParser,
        file_cleaner: FileCleaner,
        config: ConfigManager,
        job: SimulationJobSchema,
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
