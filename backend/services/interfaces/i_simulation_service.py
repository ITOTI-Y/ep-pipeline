from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

from backend.models import SimulationResult

if TYPE_CHECKING:
    from backend.models import SimulationJob
    from backend.services.simulation.executor import EnergyPlusExecutor
    from backend.services.simulation.file_cleaner import FileCleaner
    from backend.services.simulation.result_parser import ResultParser
    from backend.utils.config import ConfigManager


class ISimulationService(ABC):
    """Template for an EnergyPlus simulation stage.

    Subclasses set ``_job``, ``_config``, ``_executor``, ``_result_parser`` and
    ``_file_cleaner`` in ``__init__`` and implement ``prepare()``. The shared
    ``execute()``/``cleanup()``/``run()`` flow is provided here; override
    ``_cleanup_exclude`` to change which files survive cleanup, or ``run()`` to
    attach stage-specific fields to the result.
    """

    _job: SimulationJob
    _config: ConfigManager
    _executor: EnergyPlusExecutor
    _result_parser: ResultParser
    _file_cleaner: FileCleaner
    _cleanup_exclude: tuple[str, ...] = ("*.sql", "*.csv")

    @abstractmethod
    def prepare(self) -> None:
        """Apply stage-specific configuration to the IDF before execution."""

    def execute(self) -> SimulationResult:
        logger.info(f"Executing simulation for job {self._job.id}")
        result = SimulationResult(
            job_id=self._job.id,
            building_type=self._job.building.building_type,
        )
        try:
            result = self._executor.run(job=self._job)
            return self._result_parser.parse(result=result, job=self._job)
        except Exception as e:
            logger.exception(f"Failed to execute simulation for job {self._job.id}")
            result.add_error(str(e))
            return result

    def cleanup(self) -> None:
        self._file_cleaner.clean(
            job=self._job,
            config=self._config,
            exclude_files=self._cleanup_exclude,
        )

    def run(self) -> SimulationResult:
        try:
            self.prepare()
            return self.execute()
        finally:
            self.cleanup()
