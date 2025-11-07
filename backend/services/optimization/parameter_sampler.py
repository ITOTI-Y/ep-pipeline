import numpy as np
from loguru import logger

from backend.domain.models import BuildingType, ECMParameters
from backend.utils.config import ConfigManager


class ParameterSampler:
    def __init__(self, config: ConfigManager, seed: int = 1):
        self._logger = logger.bind(service=self.__class__.__name__)
        self._seed = seed
        np.random.seed(seed)
        self._ecm_parameters: dict[str, list] = config.ecm_parameters.model_dump()
        self._ecm_parameters_names: list[str] = config.ecm_parameters.keys

    def sample(
        self,
        n_samples: int,
        building_type: BuildingType,
    ) -> list[ECMParameters]:
        self._logger.info(
            f"Generating {n_samples} samples for building type {building_type}"
        )
        ecm_samples = []
        count = 0
        while count < n_samples:
            ecm_model = ECMParameters(building_type=building_type)
            for param_name in self._ecm_parameters_names:
                select_value = np.random.choice(self._ecm_parameters[param_name])
                ecm_model.__setattr__(param_name, select_value)
            if ecm_model in ecm_samples:
                continue
            ecm_samples.append(ecm_model)
            count += 1

        return ecm_samples
