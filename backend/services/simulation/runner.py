from collections.abc import Callable
from functools import cache
from itertools import groupby
from pathlib import Path

from idfpy import IDF
from joblib import Parallel, delayed, load
from loguru import logger

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.models import ECMParameters, SimulationJob
from backend.models.config_models import IDFFile, WeatherFile
from backend.models.simulation_job import SimulationType
from backend.services.optimization.parameter_sampler import ParameterSampler
from backend.services.optimization.surrogate_model import (
    collect_ecm_rows,
    delete_ecm_outputs,
    train_and_save_surrogate_model,
)
from backend.services.simulation._share import (
    ISimulationService,
    job_output_dir,
    job_surrogate_model_path,
)
from backend.services.simulation.baseline_service import BaselineService
from backend.services.simulation.data_extractor import DataExtractor
from backend.services.simulation.ecm_service import ECMService
from backend.services.simulation.file_cleaner import FileCleaner
from backend.services.simulation.optimization_service import OptimizationService
from backend.services.simulation.pv_service import PVService
from backend.services.simulation.result_parser import ResultParser
from backend.utils.config import ConfigManager


@cache
def _worker_config(config_dir: str) -> ConfigManager:
    return ConfigManager(Path(config_dir))


def _group_key(item):
    idf, *_ = item
    return (idf.city, idf.building_type)


def build_service(
    config: ConfigManager,
    idf_file: IDFFile,
    weather_file: WeatherFile,
    simulation_type: SimulationType,
    idx: int,
    ecm_params: ECMParameters | None,
) -> ISimulationService:
    output_directory = job_output_dir(
        config, idf_file, weather_file, idx, simulation_type
    )
    job = SimulationJob(
        idf_file=idf_file,
        idf=IDF.load(idf_file.file_path),
        weather_file=weather_file,
        simulation_type=simulation_type,
        output_directory=output_directory,
        output_prefix="eplus",
        ecm_parameters=ecm_params if ecm_params is not None else None,
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
                job_output_dir(
                    config, idf_file, weather_file, 0, SimulationType.BASELINE
                )
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


def _run_job_spec(
    config_dir: str,
    idf_file: IDFFile,
    weather_file: WeatherFile,
    simulation_type: SimulationType,
    idx: int,
    ecm_params: ECMParameters | None,
) -> None:
    config = _worker_config(config_dir)
    output_dir = job_output_dir(config, idf_file, weather_file, idx, simulation_type)
    if (output_dir / "eplusout.parquet").exists():
        logger.info(f"Skipping existing job: {output_dir}")
        return
    service = build_service(
        config, idf_file, weather_file, simulation_type, idx, ecm_params
    )
    result = service.run()
    if result.success:
        logger.info(f"Job completed successfully: {output_dir}")
    else:
        logger.error(f"Job failed: {output_dir}")
    return


def parallel_run(
    simulation_type: SimulationType,
    jobs: list[tuple[IDFFile, WeatherFile, int, ECMParameters]]
    | list[tuple[IDFFile, WeatherFile, int, None]],
    n_jobs: int,
    init_worker: Callable,
    config: ConfigManager,
    config_dir: str,
) -> None:
    logger.info(f"Running {simulation_type.value} simulation")
    if simulation_type == SimulationType.ECM:
        samples_by_bt = {
            bt: ParameterSampler(config).sample(n_samples=512, building_type=bt)
            for bt in {idf.building_type for idf, _, _, _ in jobs}
        }
        for (city, building_type), group in groupby(jobs, key=_group_key):
            model_path = job_surrogate_model_path(config, city, building_type)
            if model_path.exists():
                logger.info(f"skip {city}/{building_type}: surrogate model exists")
                continue

            group_base = list(group)
            ecm_jobs = [
                (idf_file, weather_file, idx, ecm_params)
                for idf_file, weather_file, _, _ in group_base
                for idx, ecm_params in enumerate(samples_by_bt[building_type])
            ]
            _ = Parallel(
                n_jobs=n_jobs,
                verbose=10,
                backend="loky",
                initializer=init_worker,
                initargs=(str(config.paths.log_dir),),
            )(
                delayed(_run_job_spec)(
                    config_dir, idf_file, weather_file, simulation_type, idx, ecm_params
                )
                for idf_file, weather_file, idx, ecm_params in ecm_jobs
            )
            df = collect_ecm_rows(config, city, building_type)
            train_and_save_surrogate_model(config, city, building_type, df)
            delete_ecm_outputs(config, city, building_type)
    else:
        _ = Parallel(
            n_jobs=n_jobs,
            verbose=10,
            backend="loky",
            initializer=init_worker,
            initargs=(str(config.paths.log_dir),),
        )(
            delayed(_run_job_spec)(
                config_dir, idf_file, weather_file, simulation_type, idx, ecm_params
            )
            for idf_file, weather_file, idx, ecm_params in jobs
        )


__all__ = [
    "_run_job_spec",
    "_worker_config",
]
