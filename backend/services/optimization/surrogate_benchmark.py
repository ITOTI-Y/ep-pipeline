import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from backend.models.config_models import ECMParametersConfigSchema
from backend.utils.config import ConfigManager

FEATURE_NAMES: Final = ECMParametersConfigSchema().keys
TARGET_NAMES: Final = [
    "net_site_eui",
    "net_source_eui",
    "total_site_eui",
    "total_source_eui",
]

_N_ESTIMATORS: Final = 300
_LEARNING_RATE: Final = 0.1
_MAX_DEPTH: Final = 6


@dataclass
class ModelSpec:
    estimator: Any
    multioutput: str
    scaled: bool


def build_registry(seed: int) -> dict[str, ModelSpec]:
    return {
        "Ridge": ModelSpec(
            make_pipeline(StandardScaler(), Ridge(random_state=seed)),
            "native",
            scaled=True,
        ),
        "KNN": ModelSpec(
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            "native",
            scaled=True,
        ),
        "SVR": ModelSpec(
            make_pipeline(StandardScaler(), MultiOutputRegressor(SVR())),
            "MultiOutputRegressor",
            scaled=True,
        ),
        "MLP": ModelSpec(
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(128, 128),
                    max_iter=1000,
                    early_stopping=True,
                    n_iter_no_change=15,
                    random_state=seed,
                ),
            ),
            "native",
            scaled=True,
        ),
        "DecisionTree": ModelSpec(
            DecisionTreeRegressor(max_depth=12, random_state=seed),
            "native",
            False,
        ),
        "RandomForest": ModelSpec(
            RandomForestRegressor(n_estimators=_N_ESTIMATORS, random_state=seed),
            "native",
            False,
        ),
        "ExtraTrees": ModelSpec(
            ExtraTreesRegressor(n_estimators=_N_ESTIMATORS, random_state=seed),
            "native",
            False,
        ),
        "GradientBoosting": ModelSpec(
            MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=_N_ESTIMATORS,
                    learning_rate=_LEARNING_RATE,
                    max_depth=_MAX_DEPTH,
                    random_state=seed,
                )
            ),
            "MultiOutputRegressor",
            False,
        ),
        "LightGBM": ModelSpec(
            MultiOutputRegressor(
                LGBMRegressor(
                    n_estimators=_N_ESTIMATORS,
                    learning_rate=_LEARNING_RATE,
                    max_depth=_MAX_DEPTH,
                    random_state=seed,
                    verbose=-1,
                )
            ),
            "MultiOutputRegressor",
            False,
        ),
        "CatBoost": ModelSpec(
            CatBoostRegressor(
                iterations=_N_ESTIMATORS,
                learning_rate=_LEARNING_RATE,
                depth=_MAX_DEPTH,
                loss_function="MultiRMSE",
                random_seed=seed,
                verbose=False,
            ),
            "MultiRMSE",
            False,
        ),
        "XGBoost": ModelSpec(
            XGBRegressor(
                random_state=seed,
                objective="reg:squarederror",
                n_estimators=_N_ESTIMATORS,
                max_depth=_MAX_DEPTH,
                learning_rate=_LEARNING_RATE,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                multi_strategy="multi_output_tree",
                eval_metric="rmse",
            ),
            "multi_output_tree",
            False,
        ),
    }


class SurrogateBenchmark:
    def __init__(
        self,
        config: ConfigManager,
        ecm_csv_path: Path | None = None,
        n_splits: int = 5,
        latency_reps: int = 50,
    ) -> None:
        self._config = config
        self._seed = config.optimization.seed
        self._n_splits = n_splits
        self._latency_reps = latency_reps
        csv_path = ecm_csv_path or config.paths.ecm_dir / "results.csv"
        self._data = pd.read_csv(csv_path)
        logger.info(f"Benchmark data loaded from {csv_path} ({len(self._data)} rows)")
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._encoder.fit(self._data["code"].values.reshape(-1, 1))
        np.random.seed(self._seed)

    def _build_xy(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        categorical = self._encoder.transform(data["code"].values.reshape(-1, 1))
        x = np.concatenate(
            [data[FEATURE_NAMES].values.astype(np.float32), categorical],
            axis=1,
        )
        y = data[TARGET_NAMES].values.astype(np.float32)
        return x, y

    def _cv_one(
        self, spec: ModelSpec, x: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        kf = KFold(n_splits=self._n_splits, shuffle=True, random_state=self._seed)
        n_targets = y.shape[1]
        r2s: list[float] = []
        rmses: list[float] = []
        maes: list[float] = []
        fit_times: list[float] = []
        latencies: list[float] = []
        per_target_r2: list[list[float]] = [[] for _ in range(n_targets)]

        for train_idx, test_idx in kf.split(x):
            est = clone(spec.estimator)
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            t0 = time.perf_counter()
            est.fit(x_train, y_train)
            fit_times.append(time.perf_counter() - t0)

            y_pred = np.asarray(est.predict(x_test))
            r2s.append(float(r2_score(y_test, y_pred)))
            rmses.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
            maes.append(float(mean_absolute_error(y_test, y_pred)))
            for i in range(n_targets):
                per_target_r2[i].append(float(r2_score(y_test[:, i], y_pred[:, i])))

            sample = x_test[:1]
            t0 = time.perf_counter()
            for _ in range(self._latency_reps):
                est.predict(sample)
            latencies.append((time.perf_counter() - t0) / self._latency_reps * 1000)

        metrics: dict[str, float] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "fit_time_s_mean": float(np.mean(fit_times)),
            "predict_latency_ms_mean": float(np.median(latencies)),
        }
        for i in range(n_targets):
            metrics[f"output_{i + 1}_r2_mean"] = float(np.mean(per_target_r2[i]))
        return metrics

    def run(self) -> pd.DataFrame:
        registry = build_registry(self._seed)
        rows: list[dict[str, Any]] = []
        for building_type, data in self._data.groupby("building_type"):
            x, y = self._build_xy(data)
            for name, spec in registry.items():
                logger.info(f"CV {name} on {building_type} (n={len(data)})")
                metrics = self._cv_one(spec, x, y)
                rows.append(
                    {
                        "building_type": str(building_type),
                        "model": name,
                        "multioutput_strategy": spec.multioutput,
                        **metrics,
                    }
                )

        df = pd.DataFrame(rows)
        numeric_cols = [
            c
            for c in df.columns
            if c not in ("building_type", "model", "multioutput_strategy")
        ]
        agg = df.groupby("model", as_index=False)[numeric_cols].mean()
        agg["building_type"] = "All"
        strategy = df.drop_duplicates("model").set_index("model")[
            "multioutput_strategy"
        ]
        agg["multioutput_strategy"] = agg["model"].map(strategy)

        return pd.concat([df, agg[df.columns]], ignore_index=True)
