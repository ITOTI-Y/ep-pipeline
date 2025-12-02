from collections.abc import Generator
from itertools import chain, product
from pathlib import Path
from pickle import dump

from eppy.modeleditor import IDF
from joblib import Parallel, cpu_count, delayed
from loguru import logger

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.models import (
    Building,
    BuildingType,
    SimulationJob,
    SimulationType,
    Weather,
)
from backend.script.parse_data import parse_results_to_csv, parse_optimal_data
from backend.services.optimization import ParameterSampler
from backend.services.simulation import (
    BaselineService,
    ECMService,
    FileCleaner,
    OptimizationService,
    ResultParser,
)
from backend.utils.config import ConfigManager, set_logger


def base_services_prepare(
    config: ConfigManager,
    buildings_weather_combinations: list[tuple[Building, Weather]],
) -> Generator[tuple[SimulationJob, BaselineService]]:
    for building, weather in buildings_weather_combinations:
        job = SimulationJob(
            building=building,
            weather=weather,
            simulation_type=SimulationType.BASELINE,
            output_directory=config.paths.baseline_dir / building.name / weather.code,  # type: ignore
            output_prefix="baseline_",
            # idf=IDF(str(building.idf_file_path)),
        )

        baseline_service = BaselineService(
            executor=EnergyPlusExecutor(),
            result_parser=ResultParser(),
            file_cleaner=FileCleaner(),
            config=config,
            job=job,
        )

        yield job, baseline_service


def ecm_services_prepare(
    config: ConfigManager,
    buildings_weather_combinations: list[tuple[Building, Weather]],
) -> Generator[tuple[SimulationJob, ECMService]]:
    n_sample = 512

    sampler = ParameterSampler(config=config)

    for building, weather in buildings_weather_combinations:
        ecm_samples = sampler.sample(
            n_samples=n_sample, building_type=building.building_type
        )
        for i, ecm_sample in enumerate(ecm_samples):
            job = SimulationJob(
                building=building,
                weather=weather,
                simulation_type=SimulationType.ECM,
                output_directory=config.paths.ecm_dir  # type: ignore
                / building.name
                / weather.code
                / f"sample_{i:03d}",
                output_prefix=f"ecm_{i:03d}",
                # idf=IDF(str(building.idf_file_path)),
                ecm_parameters=ecm_sample,
            )
            ecm_service = ECMService(
                executor=EnergyPlusExecutor(),
                result_parser=ResultParser(),
                file_cleaner=FileCleaner(),
                config=config,
                job=job,
            )
            yield job, ecm_service


def optimization_services_prepare(
    config: ConfigManager,
    buildings_weather_combinations: list[tuple[Building, Weather]],
):
    ecm_csv_path = config.paths.ecm_dir / "results.csv"
    for building, weather in buildings_weather_combinations:
        job = SimulationJob(
            building=building,
            weather=weather,
            simulation_type=SimulationType.OPTIMIZATION,
            output_directory=config.paths.optimization_dir
            / building.name
            / weather.code,  # type: ignore
            output_prefix="optimization_",
        )

        optimization_service = OptimizationService(
            executor=EnergyPlusExecutor(),
            result_parser=ResultParser(),
            file_cleaner=FileCleaner(),
            ecm_csv_path=ecm_csv_path,
            config=config,
            job=job,
        )
        yield job, optimization_service


def main():
    config = ConfigManager(Path("backend/configs"))
    set_logger(config.paths.log_dir)
    logger.info("Starting simulation")
    idf_files = config.paths.idf_files
    weather_files = config.paths.ftmy_files + config.paths.tmy_files

    buildings = []
    for idf_file in idf_files:
        building = Building(
            name=idf_file.stem,
            building_type=BuildingType.from_str(idf_file.stem),
            location="Chicago",
            idf_file_path=idf_file,
        )
        buildings.append(building)

    weathers = []
    for weather_file in weather_files:
        weather = Weather(
            file_path=weather_file,
            location="Chicago",
        )
        weathers.append(weather)

    buildings_weather_combinations = list(product(buildings, weathers))

    n_jobs = cpu_count() - 2 if cpu_count() > 2 else 1

    base_services = base_services_prepare(config, buildings_weather_combinations)
    ecm_services = ecm_services_prepare(config, buildings_weather_combinations)
    optimization_services = optimization_services_prepare(
        config, buildings_weather_combinations
    )

    _ = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_single_run)(job, service, config)
        for job, service in chain(base_services, ecm_services)
    )
    parse_results_to_csv(config)

    _ = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_single_run)(job, service, config)
        for job, service in optimization_services
    )


def _single_run(
    job: SimulationJob, service: BaselineService | ECMService, config: ConfigManager
):
    set_logger(config.paths.log_dir)
    IDF.setiddname(str(config.paths.idd_file))
    job.idf = IDF(str(job.building.idf_file_path))

    result = service.run()

    with open(job.output_directory / "result.pkl", "wb") as f:
        dump(result, f)

    return result


if __name__ == "__main__":
    # main()
    # parse_results_to_csv(ConfigManager(Path("backend/configs")))
    parse_optimal_data(ConfigManager(Path("backend/configs")))
