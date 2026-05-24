from collections import defaultdict
from itertools import chain, product  # noqa: F401
from pathlib import Path
from typing import Annotated

import typer
from joblib import Parallel, cpu_count, delayed
from loguru import logger

from backend.citys.cli import app as citys_app
from backend.models.simulation_job import BuildingWeatherCombination, SimulationType
from backend.script.gen_manifest import check as manifest_check
from backend.script.gen_manifest import generate as manifest_generate
from backend.script.gen_manifest import save as manifest_save
from backend.script.parse_data import (  # noqa: F401
    parse_optimal_data,
    parse_result_parameters,
    parse_results_to_csv,
)
from backend.services.simulation import (
    get_simulation_services,
)
from backend.services.simulation._share import ISimulationService
from backend.utils.config import ConfigManager, set_logger

app = typer.Typer()
app.add_typer(citys_app, name="city", help="City selection pipeline")


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
        _baseline_services = get_simulation_services(
            config, combination, SimulationType.BASELINE
        )
        services[SimulationType.BASELINE].extend(_baseline_services)

        # _ecm_services = get_simulation_services(config, combination, SimulationType.ECM)
        # services[SimulationType.ECM].extend(_ecm_services)

        # _optimization_services = get_simulation_services(
        #     config, combination, SimulationType.OPTIMIZATION
        # )
        # services[SimulationType.OPTIMIZATION].extend(_optimization_services)

        # _pv_services = get_simulation_services(config, combination, SimulationType.PV)
        # services[SimulationType.PV].extend(_pv_services)

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
