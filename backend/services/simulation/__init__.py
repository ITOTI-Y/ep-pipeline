from collections.abc import Generator
from itertools import product
from pickle import load

from idfpy import IDF

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.models.config_models import IDFFile, WeatherFile
from backend.models.simulation_job import (
    BuildingWeatherCombination,
    SimulationJob,
    SimulationType,
)
from backend.services.simulation._share import ISimulationService
from backend.services.simulation.baseline_service import BaselineService
from backend.services.simulation.data_extractor import DataExtractor
from backend.services.simulation.ecm_service import ECMService
from backend.services.simulation.file_cleaner import FileCleaner
from backend.services.simulation.optimization_service import OptimizationService
from backend.services.simulation.result_parser import ResultParser
from backend.utils.config import ConfigManager

from .pv_service import PVService


def build_service(
    config: ConfigManager,
    idf_file: IDFFile,
    weather_file: WeatherFile,
    simulation_type: SimulationType,
) -> ISimulationService:
    output_directory = (
        config.paths.sim_dir
        / simulation_type.value
        / weather_file.code
        / idf_file.city
        / idf_file.building_type
    )
    job = SimulationJob(
        idf_file=idf_file,
        idf=IDF.load(idf_file.file_path),
        weather_file=weather_file,
        simulation_type=simulation_type,
        output_directory=output_directory,
        output_prefix="eplus",
    )

    match simulation_type:
        case SimulationType.BASELINE:
            return BaselineService(
                executor=EnergyPlusExecutor(),
                result_parser=ResultParser(),
                file_cleaner=FileCleaner(),
                data_extractor=DataExtractor(),
                config=config,
                job=job,
            )
        case SimulationType.ECM:
            return ECMService(
                executor=EnergyPlusExecutor(),
                result_parser=ResultParser(),
                file_cleaner=FileCleaner(),
                data_extractor=DataExtractor(),
                config=config,
                job=job,
            )
        case SimulationType.OPTIMIZATION:
            return OptimizationService(
                executor=EnergyPlusExecutor(),
                result_parser=ResultParser(),
                file_cleaner=FileCleaner(),
                data_extractor=DataExtractor(),
                config=config,
                job=job,
            )
        case SimulationType.PV:
            baseline_result_path = (
                config.paths.sim_dir
                / SimulationType.BASELINE.value
                / weather_file.code
                / idf_file.city
                / idf_file.building_type
                / "result.pkl"
            )
            with open(baseline_result_path, "rb") as f:
                baseline_result = load(f)
            return PVService(
                executor=EnergyPlusExecutor(),
                result_parser=ResultParser(),
                file_cleaner=FileCleaner(),
                data_extractor=DataExtractor(),
                config=config,
                job=job,
                surfaces=baseline_result.surfaces,
            )
        case _:
            raise ValueError(f"Invalid simulation type: {simulation_type}")


def get_simulation_services(
    config: ConfigManager,
    combination: BuildingWeatherCombination,
    simulation_type: SimulationType,
) -> Generator[ISimulationService]:
    for idf_file, weather_file in product(
        combination.idf_files, combination.weather_files
    ):
        yield build_service(config, idf_file, weather_file, simulation_type)


__all__ = [
    "FileCleaner",
    "ResultParser",
    "build_service",
    "get_simulation_services",
]
