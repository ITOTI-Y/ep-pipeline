import json
from pathlib import Path
from pickle import dump, load

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import OneHotEncoder

from backend.models import SimulationJob, SimulationResult
from backend.services.configuration import (
    ECMApply,
    OutputApply,
    PeriodApply,
    SettingApply,
)
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IFileCleaner,
    IResultParser,
    ISimulationService,
)
from backend.services.optimization.optimization_model import GeneticAlgorithmModel
from backend.services.optimization.surrogate_model import (
    ISurrogateModel,
    XGBoostSurrogateModel,
)
from backend.utils.config import ConfigManager

FEATURE_NAMES = [
    "window_shgc",
    "window_u_value",
    "visible_transmittance",
    "wall_insulation",
    "infiltration_rate",
    "natural_ventilation_area",
    "cooling_cop",
    "heating_cop",
    "cooling_air_temperature",
    "heating_air_temperature",
    "lighting_power_reduction_level",
]

TARGET_NAMES = [
    "net_site_eui",
    "net_source_eui",
    "total_site_eui",
    "total_source_eui",
]


class OptimizationService(ISimulationService):
    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: IResultParser,
        file_cleaner: IFileCleaner,
        config: ConfigManager,
        job: SimulationJob,
        ecm_csv_path: Path | None = None,
    ):
        np.random.seed(config.optimization.seed)
        self._ecm_csv_path = ecm_csv_path or config.paths.ecm_dir / "results.csv"
        self._ecm_data = pd.read_csv(self._ecm_csv_path)
        logger.info(f"ECM data loaded from {self._ecm_csv_path}")
        self._ecm_parameters_names = config.ecm_parameters.keys
        self._config = config
        self._job = job
        self._surrogate_model = None
        self._one_hot_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self._prepare_surrogate_models()
        self._ecm_apply = ECMApply()
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)
        self._setting_apply = SettingApply(config=config)
        self._executor = executor
        self._result_parser = result_parser
        self._file_cleaner = file_cleaner

        self._predicted_eui = None

    def _prepare_surrogate_models(self) -> None:
        surrogate_model_path = (
            self._config.paths.optimization_dir
            / self._job.building.name
            / "surrogate_model.pkl"
        )
        encode_model_path = self._config.paths.optimization_dir / "encode_model.pkl"
        if not encode_model_path.exists():
            self._one_hot_encoder.fit(self._ecm_data["code"].values.reshape(-1, 1))  # type: ignore
            self._save_encode_model(self._one_hot_encoder, encode_model_path)
        else:
            with open(encode_model_path, "rb") as f:
                self._one_hot_encoder = load(f)

        if surrogate_model_path.exists():
            with open(surrogate_model_path, "rb") as f:
                self._surrogate_model = load(f)
            return
        else:
            group_data = self._ecm_data.groupby("building_type")
            for building_type, data in group_data:
                surrogate_model = XGBoostSurrogateModel(config=self._config)
                categorical_features = self._one_hot_encoder.transform(
                    data["code"].values.reshape(-1, 1)  # type: ignore
                )
                x = np.concatenate(
                    [
                        data[FEATURE_NAMES].values.astype(np.float32),
                        categorical_features,  # type: ignore
                    ],
                    axis=1,
                )
                y = data[TARGET_NAMES].values.astype(np.float32)
                surrogate_model.train(x, y)

                evaluate_metrics = surrogate_model.evaluate()
                evaluate_file_path = surrogate_model_path.parent / "evaluate.json"
                evaluate_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(
                    file=evaluate_file_path,
                    mode="w",
                    encoding="utf-8",
                ) as f:
                    json.dump(evaluate_metrics, f, indent=4)

                if str(building_type) == self._job.building.name:
                    self._surrogate_model = surrogate_model
                logger.info(
                    f"Surrogate model trained for building type {building_type}"
                )
                self._save_surrogate_model(surrogate_model, surrogate_model_path)

    def _save_surrogate_model(
        self, surrogate_model: ISurrogateModel, surrogate_model_path: Path
    ) -> None:
        surrogate_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(surrogate_model_path, "wb") as f:
            dump(surrogate_model, f)
            logger.info(f"Surrogate model saved to {surrogate_model_path}")

    def _save_encode_model(
        self, encode_model: OneHotEncoder, encode_model_path: Path
    ) -> None:
        encode_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(encode_model_path, "wb") as f:
            dump(encode_model, f)
            logger.info(f"Encode model saved to {encode_model_path}")

    def _get_best_ecm_parameters(self):
        building_type = self._job.building.building_type
        surrogate_model = self._surrogate_model
        if surrogate_model is None:
            raise ValueError("Surrogate model not found")
        optimization_model = GeneticAlgorithmModel(
            config=self._config,
            surrogate_model=surrogate_model,
            encode_model=self._one_hot_encoder,
            code=self._job.weather.code,  # type: ignore[arg-type]
        )
        best_ecm, predicted_eui = optimization_model.optimize(
            building_type=building_type
        )
        self._predicted_eui = predicted_eui
        self._job.ecm_parameters = best_ecm

    def prepare(self) -> None:
        self._get_best_ecm_parameters()
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._ecm_apply.apply(self._job)  # type: ignore
        self._setting_apply.apply(self._job)
        logger.info("Optimization preparation completed")

    def cleanup(self) -> None:
        self._file_cleaner.clean(
            job=self._job,
            config=self._config,
            exclude_files=("*.sql", "*.csv"),
        )

    def execute(self) -> SimulationResult:
        result = SimulationResult(
            job_id=self._job.id,
            building_type=self._job.building.building_type,
        )
        try:
            result = self._executor.run(
                job=self._job,
            )
            result = self._result_parser.parse(
                result=result,
                job=self._job,
            )
            return result
        except Exception as e:
            logger.exception(
                f"Failed to execute optimization simulation for job {self._job.id}"
            )
            result.add_error(str(e))
            return result

    def run(self) -> SimulationResult:
        try:
            self.prepare()
            result = self.execute()
            result.ecm_parameters = self._job.ecm_parameters or None
            result.predicted_eui = self._predicted_eui
            result.weather_code = self._job.weather.code
            return result
        finally:
            self.cleanup()
