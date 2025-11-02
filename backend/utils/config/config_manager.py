from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

from .config_models import AnalysisConfig, PathsConfig, SimulationConfig


class ConfigManager:
    def __init__(self, config_dir: Path = Path("backend/configs")):
        self._config_dir = config_dir
        self._raw_config = self._load_config()

        self.paths = self._parse_paths_config()
        self.simulation = self._parse_simulation_config()
        self.analysis = self._parse_analysis_config()

    def _load_config(self) -> ListConfig | DictConfig:
        config_files = sorted(self._config_dir.glob("*.yaml"))
        configs = []

        for file in config_files:
            config = OmegaConf.load(file)
            configs.append(config)

        merged_config = OmegaConf.merge(*configs)
        return merged_config

    def _parse_paths_config(self) -> PathsConfig:
        paths_dict = OmegaConf.to_container(
            self._raw_config.paths,
            resolve=True,
            throw_on_missing=True,
        )

        return PathsConfig(**paths_dict)  # type: ignore

    def _parse_simulation_config(self) -> SimulationConfig:
        sim_dict = OmegaConf.to_container(
            self._raw_config.simulation,
            resolve=True,
            throw_on_missing=True,
        )

        return SimulationConfig(**sim_dict)  # type: ignore

    def _parse_analysis_config(self) -> AnalysisConfig:
        analysis_dict = OmegaConf.to_container(
            self._raw_config.analysis,
            resolve=True,
            throw_on_missing=False,
        )

        return AnalysisConfig(**analysis_dict)  # type: ignore

    @property
    def value(self):
        return self._raw_config.copy()
