import numpy as np
from loguru import logger
from scipy.stats.qmc import LatinHypercube

from backend.models import BuildingType, ECMParameters
from backend.utils.config import ConfigManager

# class ParameterSampler:
#     def __init__(self, config: ConfigManager, seed: int = 1):
#         self._seed = seed
#         np.random.seed(seed)
#         self._ecm_parameters: dict[str, list] = config.ecm_parameters.model_dump()
#         self._ecm_parameters_names: list[str] = config.ecm_parameters.keys

#     def sample(
#         self,
#         n_samples: int,
#         building_type: BuildingType,
#     ) -> list[ECMParameters]:
#         logger.info(f"Generating {n_samples} samples for building type {building_type}")
#         ecm_samples = []
#         attempts = 0
#         max_attempts = n_samples * 10
#         while len(ecm_samples) < n_samples:
#             attempts += 1
#             if attempts > max_attempts:
#                 raise RuntimeError(
#                     "Failed to generate unique samples after max attempts"
#                 )
#             ecm_model = ECMParameters(building_type=building_type)
#             for param_name in self._ecm_parameters_names:
#                 select_value = np.random.choice(self._ecm_parameters[param_name])
#                 ecm_model.__setattr__(param_name, select_value)
#             if ecm_model in ecm_samples:
#                 continue
#             ecm_samples.append(ecm_model)

#         return ecm_samples


class ParameterSampler:
    def __init__(self, config: ConfigManager, seed: int = 1):
        self._seed = seed
        np.random.seed(seed)
        self._ecm_parameters: dict[str, list] = config.ecm_parameters.model_dump()
        self._ecm_parameters_names: list[str] = config.ecm_parameters.keys

    def sample(
        self,
        n_samples: int,
        building_type: BuildingType,
    ) -> list[ECMParameters]:
        logger.info(f"Generating {n_samples} samples for building type {building_type}")

        n_dimensions: int = len(self._ecm_parameters_names)

        lhs_sampler = LatinHypercube(d=n_dimensions)
        lhs_samples = lhs_sampler.random(n=n_samples)

        ecm_samples = []
        seen_samples = set()

        for sample_idx in range(n_samples):
            ecm_model = ECMParameters(building_type=building_type)
            param_values = []

            for dim_idx, param_name in enumerate(self._ecm_parameters_names):
                param_options = self._ecm_parameters[param_name]
                n_options = len(param_options)

                lhs_value = lhs_samples[sample_idx, dim_idx]
                option_idx = int(lhs_value * n_options)
                option_idx = min(option_idx, n_options - 1)

                selected_value = param_options[option_idx]
                ecm_model.__setattr__(param_name, selected_value)
                param_values.append(selected_value)

            sample_tuple = tuple(param_values)
            if sample_tuple in seen_samples:
                continue
            seen_samples.add(sample_tuple)
            ecm_samples.append(ecm_model)

        if len(ecm_samples) < n_samples:
            logger.warning(
                f"LHS produced {len(ecm_samples)} unique samples, "
                f"supplementing to reach {n_samples}"
            )
            self._supplement_samples(ecm_samples, n_samples, seen_samples, building_type)

        logger.success(f"Generated {len(ecm_samples)} valid ECM parameter samples")
        return ecm_samples

    def _supplement_samples(
            self,
            ecm_samples: list[ECMParameters],
            n_samples: int,
            seen_samples: set[tuple],
            building_type: BuildingType,
    ):
        attempts = 0
        max_attempts = (n_samples - len(ecm_samples)) * 10

        while len(ecm_samples) < n_samples and attempts < max_attempts:
            attempts += 1
            ecm_model = ECMParameters(building_type=building_type)
            param_values = []

            for param_name in self._ecm_parameters_names:
                param_options = self._ecm_parameters[param_name]
                select_value = np.random.choice(param_options)
                ecm_model.__setattr__(param_name, select_value)
                param_values.append(select_value)

            sample_tuple = tuple(param_values)
            if sample_tuple in seen_samples:
                continue
            seen_samples.add(sample_tuple)
            ecm_samples.append(ecm_model)

        if len(ecm_samples) < n_samples:
            raise RuntimeError(
                f"Failed to generate unique samples after {max_attempts} attempts"
            )
