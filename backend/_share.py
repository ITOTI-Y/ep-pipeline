from functools import cache
from pathlib import Path

from backend.models.config_models import IDFFile, WeatherFile
from backend.models.simulation_job import SimulationType
from backend.services.simulation import build_service
from backend.utils.config import ConfigManager


@cache
def _worker_config(config_dir: str) -> ConfigManager:
    return ConfigManager(Path(config_dir))


def _run_job_spec(
    config_dir: str,
    idf_file: IDFFile,
    weather_file: WeatherFile,
    simulation_type: SimulationType,
) -> None:
    config = _worker_config(config_dir)
    service = build_service(config, idf_file, weather_file, simulation_type)
    service.run()
    return None


__all__ = [
    "_run_job_spec",
    "_worker_config",
]
