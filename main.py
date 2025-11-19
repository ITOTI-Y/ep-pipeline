from collections.abc import Generator
from itertools import product
from pathlib import Path

from eppy.modeleditor import IDF
from joblib import Parallel, delayed

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.models import (
    BaselineContext,
    Building,
    BuildingType,
    ECMContext,
    SimulationJob,
    SimulationType,
    Weather,
)
from backend.services.simulation import (
    BaselineService,
    ECMService,
    FileCleaner,
    ResultParser,
)
from backend.utils.config import ConfigManager


def base_services_prepare(
    config: ConfigManager,
    buildings_weather_combinations: list[tuple[Building, Weather]],
) -> Generator[tuple[BaselineContext, BaselineService]]:
    for building, weather in buildings_weather_combinations:
        job = SimulationJob(
            building=building,
            weather=weather,
            simulation_type=SimulationType.BASELINE,
            output_directory=config.paths.baseline_dir / building.name,
            output_prefix="baseline_",
        )

        context = BaselineContext(
            job=job,
            idf=IDF(str(job.building.idf_file_path)),
        )
        baseline_service = BaselineService(
            executor=EnergyPlusExecutor(),
            result_parser=ResultParser(),
            file_cleaner=FileCleaner(),
            config=config,
        )

        yield context, baseline_service


def ecm_services_prepare(
    config: ConfigManager,
    buildings_weather_combinations: list[tuple[Building, Weather]],
) -> Generator[tuple[ECMContext, ECMService]]:
    pass


def main():
    config = ConfigManager(Path("backend/configs"))
    idf_files = config.paths.idf_files
    weather_files = config.paths.ftmy_files

    IDF.setiddname(str(config.paths.idd_file))

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

    services = base_services_prepare(config, buildings_weather_combinations)

    results = Parallel(n_jobs=2, verbose=10, backend="loky")(
        delayed(_single_run)(context, service, config) for context, service in services
    )

    print(results)


def _single_run(context, service, config):
    IDF.setiddname(str(config.paths.idd_file))

    if isinstance(context, BaselineContext):
        result = service.run(context)
    elif isinstance(context, ECMContext):
        result = service.run(context, context.job.ecm_parameters)  # type: ignore

    return result


if __name__ == "__main__":
    main()
