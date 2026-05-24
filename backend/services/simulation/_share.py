from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pickle import dump

from backend.models import SimulationJob, SimulationResult
from backend.services.interfaces import IEnergyPlusExecutor, IResultParser
from backend.utils.config import ConfigManager


class ISimulationService(ABC):
    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: IResultParser,
        file_cleaner: IFileCleaner,
        config: ConfigManager,
        job: SimulationJob,
    ):
        self._executor = executor
        self._result_parser = result_parser
        self._file_cleaner = file_cleaner
        self._config = config
        self._job = job

    @abstractmethod
    def prepare(self) -> None:
        """
        Prepare the simulation context.

        include:
            - create output directory
            - validate files existence
            - setting output variables
            - apply preparation logic
        Raises:
            ValidationError: If validation fails.
            FileNotFoundError: If required files are missing.
            PreparationError: If preparation process fails.
        """
        pass

    @abstractmethod
    def execute(self) -> SimulationResult:
        """
        Execute the simulation.

        Returns:
            SimulationResult: The simulation result containing output paths,
                energy metrics, and execution metadata.

        Raises:
            SimulationError: If the simulation execution fails.
            RuntimeError: If EnergyPlus encounters a runtime error.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up temporary files and resources after simulation.

        This method should remove intermediate files and release any
        resources held during the simulation. It should not raise exceptions.

        """
        pass

    def run(self) -> SimulationResult:
        try:
            self.prepare()
            self._copy_schedules()
            result = self.execute()
            with open(self._job.output_directory / "result.pkl", "wb") as f:
                dump(result, f)
            return result
        finally:
            self.cleanup()

    def _copy_schedules(self) -> None:
        schedules_src = self._job.idf_file.file_path.parent / "schedules"
        schedules_dst = self._job.output_directory / "schedules"
        if schedules_src.is_dir():
            shutil.copytree(schedules_src, schedules_dst, dirs_exist_ok=True)


class IFileCleaner(ABC):
    @abstractmethod
    def clean(
        self,
        job: SimulationJob,
        config: ConfigManager,
        exclude_files: tuple[str, ...] = (),
    ) -> None:
        pass
