from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from backend.models.config_models import (
    AnalysisConfig,
    ECMParametersConfig,
    PathsConfig,
    SimulationConfig,
)


class ConfigManager:
    def __init__(self, config_dir: Path = Path("backend/configs")):
        self._logger = logger.bind(module=self.__class__.__name__)
        self._config_dir = Path(config_dir)
        self._raw_config = self._load_config()

        self.paths = self._parse_paths_config()
        self.simulation = self._parse_simulation_config()
        self.analysis = self._parse_analysis_config()
        self.ecm_parameters = self._parse_ecm_parameters_config()

    def _load_config(self) -> ListConfig | DictConfig:
        if not self._config_dir.exists():
            self._logger.error(f"Config directory not found: {self._config_dir}")
            raise FileNotFoundError(f"Config directory not found: {self._config_dir}")
        if not self._config_dir.is_dir():
            self._logger.error(f"Config path is not a directory: {self._config_dir}")
            raise NotADirectoryError(
                f"Config path is not a directory: {self._config_dir}"
            )

        config_files = sorted(self._config_dir.glob("*.yaml"))
        configs = []

        if not config_files:
            self._logger.error(f"No config files found in {self._config_dir}")
            raise FileNotFoundError(f"No config files found in {self._config_dir}")

        for file in config_files:
            config = OmegaConf.load(file)
            configs.append(config)

        merged_config = OmegaConf.merge(*configs)
        return merged_config

    def _parse_paths_config(self) -> PathsConfig:
        paths_config = OmegaConf.select(self._raw_config, "paths")
        if paths_config is None:
            self._logger.error(f"Paths config not found in {self._raw_config}")
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
            self._logger.error(f"Simulation config not found in {self._raw_config}")
            raise ValueError(f"Simulation config not found in {self._raw_config}")

        sim_dict = OmegaConf.to_container(
            simulation_config,
            resolve=True,
            throw_on_missing=True,
        )

        return SimulationConfig(**sim_dict)  # type: ignore

    def _parse_analysis_config(self) -> AnalysisConfig:
        analysis_config = OmegaConf.select(self._raw_config, "analysis")
        if analysis_config is None:
            self._logger.warning("Analysis config not found; using defaults")
            return AnalysisConfig()

        analysis_dict = OmegaConf.to_container(
            analysis_config,
            resolve=True,
            throw_on_missing=False,
        )

        return AnalysisConfig(**analysis_dict)  # type: ignore

    def _parse_ecm_parameters_config(self) -> ECMParametersConfig:
        ecm_parameters_config = OmegaConf.select(self._raw_config, "ecm_parameters")
        n_samples = OmegaConf.select(self._raw_config, "n_samples")
        if ecm_parameters_config is None:
            self._logger.warning("ECM parameters config not found; using defaults")
            return ECMParametersConfig()

        ecm_parameters_dict = OmegaConf.to_container(
            ecm_parameters_config,
            resolve=True,
            throw_on_missing=False,
        )

        return ECMParametersConfig(**ecm_parameters_dict)  # type: ignore

    @property
    def value(self):
        return self._raw_config.copy()
