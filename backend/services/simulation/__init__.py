from collections.abc import Generator
from itertools import product
from pickle import load

from idfpy import IDF

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.models.simulation_job import (
    BuildingWeatherCombination,
    SimulationJob,
    SimulationType,
)
from backend.services.simulation._share import ISimulationService
from backend.services.simulation.file_cleaner import FileCleaner
from backend.services.simulation.result_parser import ResultParser
from backend.utils.config import ConfigManager

from .baseline_service import BaselineService
from .ecm_service import ECMService
from .optimization_service import OptimizationService
from .pv_service import PVService


def get_simulation_services(
    config: ConfigManager,
    combination: BuildingWeatherCombination,
    simulation_type: SimulationType,
) -> Generator[ISimulationService]:
    jobs: list[SimulationJob] = []
    for idf_file, weather_file in product(
        combination.idf_files, combination.weather_files
    ):
        output_directory = (
            config.paths.sim_dir
            / simulation_type.value
            / weather_file.code
            / idf_file.city
            / idf_file.building_type
        )
        jobs.append(
            SimulationJob(
                idf_file=idf_file,
                idf=IDF.load(idf_file.file_path),
                weather_file=weather_file,
                simulation_type=simulation_type,
                output_directory=output_directory,
                output_prefix="eplus",
            )
        )
    match simulation_type:
        case SimulationType.BASELINE:
            for job in jobs:
                yield BaselineService(
                    executor=EnergyPlusExecutor(),
                    result_parser=ResultParser(),
                    file_cleaner=FileCleaner(),
                    config=config,
                    job=job,
                )
        case SimulationType.ECM:
            for job in jobs:
                yield ECMService(
                    executor=EnergyPlusExecutor(),
                    result_parser=ResultParser(),
                    file_cleaner=FileCleaner(),
                    config=config,
                    job=job,
                )
        case SimulationType.OPTIMIZATION:
            for job in jobs:
                yield OptimizationService(
                    executor=EnergyPlusExecutor(),
                    result_parser=ResultParser(),
                    file_cleaner=FileCleaner(),
                    config=config,
                    job=job,
                )
        case SimulationType.PV:
            for job in jobs:
                baseline_result_path = (
                    config.paths.sim_dir
                    / SimulationType.BASELINE.value
                    / job.weather_file.code
                    / job.idf_file.city
                    / job.idf_file.building_type
                    / "result.pkl"
                )
                with open(baseline_result_path, "rb") as f:
                    baseline_result = load(f)
                surfaces = baseline_result.surfaces
                yield PVService(
                    executor=EnergyPlusExecutor(),
                    result_parser=ResultParser(),
                    file_cleaner=FileCleaner(),
                    config=config,
                    job=job,
                    surfaces=surfaces,
                )
        case _:
            raise ValueError(f"Invalid simulation type: {simulation_type}")


__all__ = [
    "FileCleaner",
    "ResultParser",
    "get_simulation_services",
]
