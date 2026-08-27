from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Annotated

import typer
from joblib import cpu_count
from loguru import logger

from backend._share import parallel_run
from backend.citys.cli import app as citys_app
from backend.models.simulation_job import BuildingWeatherCombination, SimulationType
from backend.script.gen_manifest import check as manifest_check
from backend.script.gen_manifest import generate as manifest_generate
from backend.script.gen_manifest import save as manifest_save
from backend.script.parse_data import (  # noqa: F401
    parse_result_parameters,
    parse_results_to_csv,
)
from backend.utils.config import ConfigManager, set_logger

app = typer.Typer()
app.add_typer(citys_app, name="city", help="City selection pipeline")

SSP_ORDER = {
    "tmy": 0,
    "ssp126": 1,
    "ssp245": 2,
    "ssp370": 3,
    "ssp434": 4,
    "ssp585": 5,
}


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
    def _init_worker(log_dir: str) -> None:
        set_logger(Path(log_dir))

    config_dir = "backend/configs"
    config = ConfigManager(Path(config_dir))
    set_logger(config.paths.log_dir)
    logger.info("Starting simulation")
    idf_epw_map: dict[int, BuildingWeatherCombination] = defaultdict(
        lambda: BuildingWeatherCombination(set(), set())
    )

    def _get_mapping(config: ConfigManager) -> dict[int, BuildingWeatherCombination]:
        import pandas as pd

        from backend.citys._share import CitysFileName

        df = pd.read_json(config.paths.citys_dir / CitysFileName().dest_mapped_results)
        for idf_file in config.paths.idf_files:
            idf_city = idf_file.city
            if city and idf_city.lower() != city.lower():
                continue
            tmyx_wmo_id = int(
                df[df["tmyx_city"].str.lower() == idf_city.lower()][
                    "tmyx_epw_file_paths"
                ].values[0][0][-10:-4]
            )
            weather_files = [
                ftmy_file
                for ftmy_file in config.paths.ftmy_files
                if ftmy_file.wmo_id == tmyx_wmo_id
            ] + [
                tmy_file
                for tmy_file in config.paths.tmy_files
                if tmy_file.wmo_id == tmyx_wmo_id
            ]
            idf_epw_map[tmyx_wmo_id].idf_files.add(idf_file)
            idf_epw_map[tmyx_wmo_id].weather_files.update(weather_files)
        return idf_epw_map

    idf_epw_map = _get_mapping(config)

    simulation_types = (
        # SimulationType.BASELINE,
        SimulationType.ECM,
        # SimulationType.OPTIMIZATION,
        # SimulationType.PV,
    )

    jobs = sorted(
        (
            (idf_file, weather_file, 0, None)
            for combination in idf_epw_map.values()
            for idf_file, weather_file in product(
                combination.idf_files, combination.weather_files
            )
        ),
        key=lambda pair: (
            pair[0].city,
            pair[0].building_type,
            SSP_ORDER.get(pair[1].code, 99),
        ),
    )

    for sim_type in simulation_types:
        parallel_run(sim_type, jobs, n_jobs, _init_worker, config, config_dir)


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
