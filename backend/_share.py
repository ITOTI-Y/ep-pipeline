from collections.abc import Callable
from functools import cache
from itertools import groupby
from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger

from backend.models import ECMParameters
from backend.models.config_models import IDFFile, WeatherFile
from backend.models.simulation_job import SimulationType
from backend.services.optimization.parameter_sampler import ParameterSampler
from backend.services.optimization.surrogate_model import (
    collect_ecm_rows,
    delete_ecm_outputs,
    train_and_save_surrogate_model,
)
from backend.services.simulation import (
    build_service,
    job_output_dir,
    job_surrogate_model_path,
)
from backend.utils.config import ConfigManager


@cache
def _worker_config(config_dir: str) -> ConfigManager:
    return ConfigManager(Path(config_dir))


def _group_key(item):
    idf, *_ = item
    return (idf.city, idf.building_type)


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
        return None
    service = build_service(
        config, idf_file, weather_file, simulation_type, idx, ecm_params
    )
    result = service.run()
    if result.success:
        logger.info(f"Job completed successfully: {output_dir}")
    else:
        logger.error(f"Job failed: {output_dir}")
    return None


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
            bt: ParameterSampler(config).sample(n_samples=2, building_type=bt)
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
