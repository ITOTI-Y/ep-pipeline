from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from pickle import dump

from loguru import logger

from backend.models import SimulationJob, SimulationResult
from backend.models.config_models import IDFFile, WeatherFile
from backend.models.simulation_job import SimulationType
from backend.services.interfaces import IEnergyPlusExecutor, IResultParser
from backend.utils.config import ConfigManager


def job_output_dir(
    config: ConfigManager,
    idf_file: IDFFile,
    weather_file: WeatherFile,
    idx: int,
    simulation_type: SimulationType,
) -> Path:
    return (
        config.paths.sim_dir
        / simulation_type.value
        / idf_file.city
        / idf_file.building_type
        / weather_file.code
        / f"idx_{idx:03d}"
    )


def job_surrogate_model_path(
    config: ConfigManager,
    city: str,
    building_type: str,
) -> Path:
    return (
        config.paths.sim_dir
        / SimulationType.OPTIMIZATION.value
        / city
        / building_type
        / "surrogate_model.pkl"
    )


class ISimulationService(ABC):
    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: IResultParser,
        file_cleaner: IFileCleaner,
        data_extractor: IDataExtractor,
        config: ConfigManager,
        job: SimulationJob,
    ):
        self._executor = executor
        self._result_parser = result_parser
        self._file_cleaner = file_cleaner
        self._config = config
        self._job = job
        self._data_extractor = data_extractor
        self._cleanup_exclude: tuple[str, ...] = ()

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

    def run(self) -> SimulationResult:
        try:
            self.prepare()
            self._copy_schedules()
            result = self._execute()
            result.ecm_parameters = self._job.ecm_parameters
            result.weather_code = self._job.weather_file.code
            if result.success:
                if self._job.simulation_type != SimulationType.ECM:
                    self._extract_data(result)
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

    def _extract_data(self, result: SimulationResult) -> None:
        try:
            self._data_extractor.extract(self._job)
        except Exception as e:
            self._cleanup_exclude += ("*.sql",)
            logger.exception(f"Data extraction failed: {e}")
            result.add_error(f"Data extraction failed: {e}")


class IFileCleaner(ABC):
    @abstractmethod
    def clean(
        self,
        job: SimulationJob,
        config: ConfigManager,
        exclude_files: tuple[str, ...] = (),
    ) -> None:
        pass


class IDataExtractor(ABC):
    @abstractmethod
    def extract(self, job: SimulationJob) -> Path:
        pass
