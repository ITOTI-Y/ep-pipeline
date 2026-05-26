from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pickle import dump

from loguru import logger

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

    def _execute(self) -> SimulationResult:
        logger.info(
            f"Executing simulation for {self._job.simulation_type.value} job City: {self._job.idf_file.city} Building Type: {self._job.idf_file.building_type}"
        )

        result = SimulationResult(
            job_id=self._job.id,
            building_type=self._job.idf_file.building_type,
        )

        try:
            result = self._executor.run(
                job=self._job,
            )
            if result.success:
                result = self._result_parser.parse(
                    result=result,
                    job=self._job,
                )
            return result
        except Exception as e:
            logger.exception(
                f"Failed to execute simulation for {self._job.simulation_type.value} job City: {self._job.idf_file.city} Building Type: {self._job.idf_file.building_type}"
            )
            result.add_error(str(e))
            return result

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
            result = self._execute()
            if result.success:
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
