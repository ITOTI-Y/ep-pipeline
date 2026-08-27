import json
import shutil
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import dump, load
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from backend._share import FEATURE_NAMES, SSP_ORDER, TARGET_NAMES
from backend.models.simulation_job import SimulationResult, SimulationType
from backend.services.simulation._share import job_surrogate_model_path
from backend.utils.config import ConfigManager


class ISurrogateModel(ABC):
    @abstractmethod
    def __init__(self, config: ConfigManager) -> None:
        pass

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class XGBoostSurrogateModel(ISurrogateModel):
    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._seed = config.optimization.seed
        self._model = XGBRegressor(
            random_state=self._seed,
            objective="reg:squarederror",
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            multi_strategy="multi_output_tree",
            eval_metric="rmse",
            # early_stopping_rounds=10,
        )
        self._x_test: np.ndarray = np.array([])
        self._y_test: np.ndarray = np.array([])

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self._seed
        )
        self._model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            verbose=False,
        )
        self._x_test = x_test
        self._y_test = y_test

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def evaluate(self) -> dict[str, float]:
        if self._x_test.size == 0 or self._y_test.size == 0:
            logger.error("Test data not set")
            return {}

        y_pred = self.predict(self._x_test)
        r2 = r2_score(self._y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self._y_test, y_pred))
        mae = mean_absolute_error(self._y_test, y_pred)

        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
        }

        if self._y_test.ndim > 1 and self._y_test.shape[1] > 1:
            for i in range(self._y_test.shape[1]):
                r2_i = r2_score(self._y_test[:, i], y_pred[:, i])
                rmse_i = np.sqrt(mean_squared_error(self._y_test[:, i], y_pred[:, i]))
                mae_i = mean_absolute_error(self._y_test[:, i], y_pred[:, i])

                metrics[f"output_{i + 1}_r2_score"] = float(r2_i)
                metrics[f"output_{i + 1}_rmse"] = float(rmse_i)
                metrics[f"output_{i + 1}_mae"] = float(mae_i)

        return metrics


class CatboostSurrogateModel(ISurrogateModel):
    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._seed = config.optimization.seed
        self._model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            loss_function="MultiRMSE",
            random_seed=self._seed,
            verbose=False,
        )
        self._x_test: np.ndarray = np.array([])
        self._y_test: np.ndarray = np.array([])

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self._seed
        )
        self._model.fit(x_train, y_train)
        self._x_test = x_test
        self._y_test = y_test

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(self._model.predict(x))

    def evaluate(self) -> dict[str, float]:
        if self._x_test.size == 0 or self._y_test.size == 0:
            logger.error("Test data not set")
            return {}

        y_pred = self.predict(self._x_test)
        r2 = r2_score(self._y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self._y_test, y_pred))
        mae = mean_absolute_error(self._y_test, y_pred)

        metrics = {"r2": r2, "rmse": rmse, "mae": mae}

        if self._y_test.ndim > 1 and self._y_test.shape[1] > 1:
            for i in range(self._y_test.shape[1]):
                r2_i = r2_score(self._y_test[:, i], y_pred[:, i])
                rmse_i = np.sqrt(mean_squared_error(self._y_test[:, i], y_pred[:, i]))
                mae_i = mean_absolute_error(self._y_test[:, i], y_pred[:, i])
                metrics[f"output_{i + 1}_r2_score"] = float(r2_i)
                metrics[f"output_{i + 1}_rmse"] = float(rmse_i)
                metrics[f"output_{i + 1}_mae"] = float(mae_i)

        return metrics


def collect_ecm_rows(
    config: ConfigManager, city: str, building_type: str
) -> pd.DataFrame:
    rows = []
    root = config.paths.sim_dir / SimulationType.ECM.value / city / building_type
    for pkl in root.glob("**/result.pkl"):
        result = load(pkl)
        result = SimulationResult.model_validate(result)
        if not result.success or result.ecm_parameters is None:
            continue
        rows.append(
            {
                "code": result.weather_code,
                **result.ecm_parameters.model_dump(),
                **result.get_eui_summary(),
            }
        )
    return pd.DataFrame(rows)


def train_and_save_surrogate_model(
    config: ConfigManager, city: str, building_type: str, ecm_data: pd.DataFrame
) -> None:

    surrogate_model_path = job_surrogate_model_path(config, city, building_type)

    encode_model_path = surrogate_model_path.parents[3] / "encode_model.pkl"
    if not encode_model_path.exists():
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(np.array(list(SSP_ORDER.keys())).reshape(-1, 1))
        encode_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(encode_model_path, "wb") as f:
            dump(encoder, f)
            logger.info(f"Encode model saved to {encode_model_path}")
    else:
        with open(encode_model_path, "rb") as f:
            encoder = load(f)

    if surrogate_model_path.exists():
        logger.info(f"Surrogate model already exists at {surrogate_model_path}")
        return
    else:
        if len(ecm_data) < 2:
            logger.warning(
                f"Skipping {building_type}: insufficient samples ({len(ecm_data)})"
            )
            return
        surrogate_model = CatboostSurrogateModel(config)
        categorical_features = encoder.transform(
            ecm_data["code"].to_numpy().reshape(-1, 1)
        )
        x = np.concatenate(
            [
                ecm_data[FEATURE_NAMES].values.astype(np.float32),
                categorical_features,
            ],
            axis=1,
        )
        y = ecm_data[TARGET_NAMES].values.astype(np.float32)
        surrogate_model.train(x, y)

        evaluate_file_path = surrogate_model_path.parent / "evaluate.json"
        evaluate_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(
            file=evaluate_file_path,
            mode="w",
            encoding="utf-8",
        ) as f:
            evaluate_metrics = surrogate_model.evaluate()
            json.dump(evaluate_metrics, f, indent=4)

        with open(surrogate_model_path, "wb") as f:
            dump(surrogate_model, f)
            logger.info(f"Surrogate model saved to {surrogate_model_path}")


def delete_ecm_outputs(config: ConfigManager, city: str, building_type: str) -> None:
    ecm_output_dir = (
        config.paths.sim_dir / SimulationType.ECM.value / city / building_type
    )
    shutil.rmtree(ecm_output_dir)
