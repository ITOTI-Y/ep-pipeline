from collections import defaultdict
from collections.abc import Generator
from copy import deepcopy
from itertools import chain, product  # noqa: F401
from pathlib import Path
from pickle import load
from typing import Annotated

import typer
from joblib import Parallel, cpu_count, delayed
from loguru import logger

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.citys.cli import app as citys_app
from backend.models import (
    Building,
    SimulationJob,
    Weather,
)
from backend.models.simulation_job import BuildingWeatherCombination, SimulationType
from backend.script.gen_manifest import check as manifest_check
from backend.script.gen_manifest import generate as manifest_generate
from backend.script.gen_manifest import save as manifest_save
from backend.script.parse_data import (  # noqa: F401
    parse_optimal_data,
    parse_result_parameters,
    parse_results_to_csv,
)
from backend.services.optimization import ParameterSampler
from backend.services.simulation import (
    BaselineService,
    ECMService,
    FileCleaner,
    OptimizationService,
    PVService,
    ResultParser,
    get_simulation_services,
)
from backend.services.simulation._share import ISimulationService
from backend.utils.config import ConfigManager, set_logger

app = typer.Typer()
app.add_typer(citys_app, name="city", help="City selection pipeline")


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


def pv_services_prepare(
    config: ConfigManager,
    buildings_weather_combinations: list[tuple[Building, Weather]],
):
    baseline_dir = config.paths.baseline_dir
    for building, weather in buildings_weather_combinations:
        b = deepcopy(building)
        b.idf_file_path = (
            config.paths.optimization_dir  # type: ignore
            / b.name
            / weather.code
            / "optimization_.idf"
        )
        job = SimulationJob(
            building=b,
            weather=weather,
            simulation_type=SimulationType.PV,
            output_directory=config.paths.pv_dir / building.name / weather.code,  # type: ignore
            output_prefix="pv_",
        )

        baseline_result_path = (
            baseline_dir / building.name / weather.code / "result.pkl"  # type: ignore
        )
        with open(baseline_result_path, "rb") as f:
            baseline_result = load(f)

        surfaces = baseline_result.surfaces

        pv_service = PVService(
            executor=EnergyPlusExecutor(),
            result_parser=ResultParser(),
            file_cleaner=FileCleaner(),
            config=config,
            job=job,
            surfaces=surfaces,
        )
        yield job, pv_service


@app.command()
def simulate(
    city: Annotated[
        str | None, typer.Option(help="The city to run the simulation for")
    ] = None,
    n_jobs: Annotated[
        int,
        typer.Option("--n-jobs", "-n", help="The number of jobs to run in parallel"),
    ] = cpu_count() - 2 if cpu_count() > 2 else 1,
):
    config = ConfigManager(Path("backend/configs"))
    set_logger(config.paths.log_dir)
    logger.info("Starting simulation")
    idf_epw_map: dict[str, BuildingWeatherCombination] = defaultdict(
        lambda: BuildingWeatherCombination(set(), set())
    )

    for idf_file in config.paths.idf_files:
        for weather_file in config.paths.ftmy_files + config.paths.tmy_files:
            if idf_file.city == weather_file.city and (
                city is None or idf_file.city == city.lower()
            ):
                idf_epw_map[idf_file.city].idf_files.add(idf_file)
                idf_epw_map[idf_file.city].weather_files.add(weather_file)

    services: dict[SimulationType, list[ISimulationService]] = defaultdict(list)
    for _city, combination in idf_epw_map.items():
        _basline_services = get_simulation_services(
            config, combination, SimulationType.BASELINE
        )
        _ecm_services = get_simulation_services(config, combination, SimulationType.ECM)
        _optimization_services = get_simulation_services(
            config, combination, SimulationType.OPTIMIZATION
        )
        _pv_services = get_simulation_services(config, combination, SimulationType.PV)
        services[SimulationType.BASELINE].extend(_basline_services)
        services[SimulationType.ECM].extend(_ecm_services)
        services[SimulationType.OPTIMIZATION].extend(_optimization_services)
        services[SimulationType.PV].extend(_pv_services)

    for _, simulation_services in services.items():
        _ = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
            delayed(service.run)() for service in simulation_services
        )


@app.command()
def visualization():
    from backend.citys.viz.results import generation_all

    generation_all(ConfigManager(Path("backend/configs")))


@app.command()
def parse_result():
    parse_result_parameters(ConfigManager(Path("backend/configs")))


@app.command()
def manifest(
    check: Annotated[
        bool,
        typer.Option("--check", help="Check data/output against existing manifest"),
    ] = False,
):
    """Generate or verify data/output manifest."""
    if check:
        ok = manifest_check()
        raise typer.Exit(code=0 if ok else 1)
    data = manifest_generate()
    manifest_save(data)
    print(f"Manifest saved: {len(data)} files recorded.")


if __name__ == "__main__":
    app()
