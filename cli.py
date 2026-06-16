from collections.abc import Generator
from copy import deepcopy
from itertools import chain, product  # noqa: F401
from pathlib import Path
from pickle import dump, load

from eppy.modeleditor import IDF
from joblib import Parallel, cpu_count, delayed
from loguru import logger
from typer import Typer

from backend.models import (
    BuildingSchema,
    BuildingType,
    SimulationJobSchema,
    SimulationType,
    WeatherSchema,
)
from backend.script.parse_data import (  # noqa: F401
    parse_result_parameters,
    parse_results_to_csv,
)
from backend.services.interfaces import ISimulationService
from backend.services.optimization import ParameterSampler
from backend.services.simulation import (
    BaselineService,
    ECMService,
    EnergyPlusExecutor,
    FileCleaner,
    OptimizationService,
    PVService,
    ResultParser,
)
from backend.utils.config import ConfigManager, set_logger

app = Typer()


def base_services_prepare(
    config: ConfigManager,
    buildings_weather_combinations: list[tuple[BuildingSchema, WeatherSchema]],
) -> Generator[tuple[SimulationJobSchema, BaselineService]]:
    for building, weather in buildings_weather_combinations:
        job = SimulationJobSchema(
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
    buildings_weather_combinations: list[tuple[BuildingSchema, WeatherSchema]],
) -> Generator[tuple[SimulationJobSchema, ECMService]]:
    n_sample = 512

    sampler = ParameterSampler(config=config)

    for building, weather in buildings_weather_combinations:
        ecm_samples = sampler.sample(
            n_samples=n_sample, building_type=building.building_type
        )
        for i, ecm_sample in enumerate(ecm_samples):
            job = SimulationJobSchema(
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
    buildings_weather_combinations: list[tuple[BuildingSchema, WeatherSchema]],
):
    ecm_csv_path = config.paths.ecm_dir / "results.csv"
    for building, weather in buildings_weather_combinations:
        job = SimulationJobSchema(
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
    buildings_weather_combinations: list[tuple[BuildingSchema, WeatherSchema]],
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
        job = SimulationJobSchema(
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
def simulation_all(city: str):
    def _single_run(
        job: SimulationJobSchema, service: ISimulationService, config: ConfigManager
    ):
        set_logger(config.paths.log_dir)
        IDF.setiddname(str(config.paths.idd_file))
        job.idf = IDF(str(job.building.idf_file_path))

        result = service.run()

        with open(job.output_directory / "result.pkl", "wb") as f:
            dump(result, f)

        return result

    config = ConfigManager(Path("backend/configs"))
    set_logger(config.paths.log_dir)
    logger.info("Starting simulation")
    idf_files = config.paths.idf_files
    weather_files = config.paths.ftmy_files + config.paths.tmy_files

    buildings = []
    for idf_file in idf_files:
        building = BuildingSchema(
            name=idf_file.stem,
            building_type=BuildingType.from_str(idf_file.stem),
            location="Chicago",
            idf_file_path=idf_file,
        )
        buildings.append(building)

    weathers = []
    for weather_file in weather_files:
        if city.lower() not in weather_file.stem.lower():
            continue
        weather = WeatherSchema(
            file_path=weather_file,
            location="Chicago",
        )
        weathers.append(weather)

    buildings_weather_combinations = list(product(buildings, weathers))

    n_jobs = cpu_count() - 2 if cpu_count() > 2 else 1

    base_services = base_services_prepare(config, buildings_weather_combinations)
    _ = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_single_run)(job, service, config) for job, service in base_services
    )
    ecm_services = ecm_services_prepare(config, buildings_weather_combinations)
    _ = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_single_run)(job, service, config)
        for job, service in ecm_services
    )
    parse_results_to_csv(config)

    optimization_services = optimization_services_prepare(
        config, buildings_weather_combinations
    )
    _ = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_single_run)(job, service, config)
        for job, service in optimization_services
    )

    pv_services = pv_services_prepare(config, buildings_weather_combinations)
    _ = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_single_run)(job, service, config) for job, service in pv_services
    )


@app.command()
def visualization():
    from backend.visualization.charts import ChartGenerator

    chart_generator = ChartGenerator(ConfigManager(Path("backend/configs")))
    chart_generator.generate_all()


@app.command()
def parse_result():
    parse_result_parameters(ConfigManager(Path("backend/configs")))


@app.command()
def prepare_paper_data():
    from backend.script.prepare_paper_data import main

    main()


@app.command()
def benchmark_surrogate():
    from backend.services.optimization.surrogate_benchmark import SurrogateBenchmark

    config = ConfigManager(Path("backend/configs"))
    df = SurrogateBenchmark(config).run()
    output_path = config.paths.csv_dir / "02b_surrogate_benchmark.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Surrogate benchmark written to {output_path}")


if __name__ == "__main__":
    app()
