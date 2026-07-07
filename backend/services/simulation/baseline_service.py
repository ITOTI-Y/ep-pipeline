from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration import GeneralApply, OutputApply, PeriodApply
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IResultParser,
)
from backend.services.simulation._share import (
    IDataExtractor,
    IFileCleaner,
    ISimulationService,
)
from backend.utils.config import ConfigManager


class BaselineService(ISimulationService):
    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: IResultParser,
        file_cleaner: IFileCleaner,
        data_extractor: IDataExtractor,
        config: ConfigManager,
        job: SimulationJob,
    ):
        super().__init__(
            executor, result_parser, file_cleaner, data_extractor, config, job
        )
        self._general_apply = GeneralApply(config=config)
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)
        self._cleanup_exclude += ("schedules",)

    def prepare(self) -> None:
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._general_apply.apply(self._job)
        logger.info("Baseline preparation completed successfully")

    def cleanup(self) -> None:
        self._file_cleaner.clean(
            job=self._job, config=self._config, exclude_files=self._cleanup_exclude
        )
