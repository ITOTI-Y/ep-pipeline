from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from backend.models.config_models import (
    ECMParametersConfig,
    OptimizationConfig,
    PathsConfig,
    PVConfig,
    SimulationConfig,
    StorageConfig,
)


class ConfigManager:
    def __init__(self, config_dir: Path = Path("backend/configs")):
        self._config_dir = Path(config_dir)
        self._raw_config = self._load_config()

        self.paths = self._parse_paths_config()
        self.simulation = self._parse_simulation_config()
        self.optimization = self._parse_optimization_config()
        self.ecm_parameters = self._parse_ecm_parameters_config()
        self.pv = self._parse_pv_config()
        self.storage = self._parse_storage_config()

    def _load_config(self) -> ListConfig | DictConfig:
        if not self._config_dir.exists():
            logger.error(f"Config directory not found: {self._config_dir}")
            raise FileNotFoundError(f"Config directory not found: {self._config_dir}")
        if not self._config_dir.is_dir():
            logger.error(f"Config path is not a directory: {self._config_dir}")
            raise NotADirectoryError(
                f"Config path is not a directory: {self._config_dir}"
            )

        config_files = sorted(self._config_dir.glob("*.yaml"))
        configs = []

        if not config_files:
            logger.error(f"No config files found in {self._config_dir}")
            raise FileNotFoundError(f"No config files found in {self._config_dir}")

        for file in config_files:
            config = OmegaConf.load(file)
            configs.append(config)

        merged_config = OmegaConf.merge(*configs)
        return merged_config

    def _parse_paths_config(self) -> PathsConfig:
        paths_config = OmegaConf.select(self._raw_config, "paths")
        if paths_config is None:
            logger.error(f"Paths config not found in {self._raw_config}")
            raise ValueError(f"Paths config not found in {self._raw_config}")

        paths_dict = OmegaConf.to_container(
            paths_config,
            resolve=True,
            throw_on_missing=True,
        )

        return PathsConfig(**paths_dict)  # type: ignore

    def _parse_simulation_config(self) -> SimulationConfig:
        simulation_config = OmegaConf.select(self._raw_config, "simulation")
        if simulation_config is None:
            logger.error(f"Simulation config not found in {self._raw_config}")
            raise ValueError(f"Simulation config not found in {self._raw_config}")

        sim_dict = OmegaConf.to_container(
            simulation_config,
            resolve=True,
            throw_on_missing=True,
        )

        return SimulationConfig(**sim_dict)  # type: ignore

    def _parse_optimization_config(self) -> OptimizationConfig:
        optimization_config = OmegaConf.select(self._raw_config, "optimization")
        if optimization_config is None:
            logger.warning("Optimization config not found; using defaults")
            return OptimizationConfig()

        optimization_dict = OmegaConf.to_container(
            optimization_config,
            resolve=True,
            throw_on_missing=False,
        )

        return OptimizationConfig(**optimization_dict)  # type: ignore

    def _parse_ecm_parameters_config(self) -> ECMParametersConfig:
        ecm_parameters_config = OmegaConf.select(self._raw_config, "ecm_parameters")
        if ecm_parameters_config is None:
            logger.warning("ECM parameters config not found; using defaults")
            return ECMParametersConfig()

        ecm_parameters_dict = OmegaConf.to_container(
            ecm_parameters_config,
            resolve=True,
            throw_on_missing=False,
        )

        return ECMParametersConfig(**ecm_parameters_dict)  # type: ignore

    def _parse_pv_config(self) -> PVConfig:
        pv_config = OmegaConf.select(self._raw_config, "pv")
        if pv_config is None:
            logger.warning("PV config not found; using defaults")
            return PVConfig()

        pv_dict = OmegaConf.to_container(
            pv_config,
            resolve=True,
            throw_on_missing=False,
        )

        return PVConfig(**pv_dict)  # type: ignore

    def _parse_storage_config(self) -> StorageConfig:
        storage_config = OmegaConf.select(self._raw_config, "storage")
        if storage_config is None:
            logger.warning("Storage config not found; using defaults")
            return StorageConfig()

        storage_dict = OmegaConf.to_container(
            storage_config,
            resolve=True,
            throw_on_missing=False,
        )

        return StorageConfig(**storage_dict)  # type: ignore

    @property
    def value(self):
        return self._raw_config.copy()
