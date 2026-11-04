import numpy as np
import pandas as pd
from joblib import load
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import OneHotEncoder

from backend._share import SSP_ORDER, TARGET_NAMES
from backend.models.simulation_job import SimulationType
from backend.services.optimization.surrogate_model import ISurrogateModel
from backend.utils.config import ConfigManager


class SensitivityResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    parameter: str
    s_t: float
    s_t_low: float
    s_t_high: float
    target_std: float


def total_order_indices(
    surrogate_model: ISurrogateModel,
    encoder: OneHotEncoder,
    grid: dict[str, list[float]],
    code: str,
    target_index: int,
    n_samples: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(grid)
    levels = [np.asarray(grid[name], dtype=np.float32) for name in names]
    code_features = encoder.transform([[code]]).astype(np.float32)

    def sample() -> np.ndarray:
        return np.column_stack(
            [rng.integers(0, len(level), n_samples) for level in levels]
        )

    def evaluate(indices: np.ndarray) -> np.ndarray:
        values = np.column_stack(
            [level[indices[:, i]] for i, level in enumerate(levels)]
        )
        x = np.concatenate(
            [values, np.repeat(code_features, len(values), axis=0)], axis=1
        )
        return surrogate_model.predict(x)[:, target_index]

    a, b = sample(), sample()
    y_a, y_b = evaluate(a), evaluate(b)
    boot = rng.integers(0, n_samples, (n_bootstrap, n_samples))

    def variance(rows: np.ndarray) -> float:
        return float(np.var(np.concatenate([y_a[rows], y_b[rows]])))

    total_variance = variance(np.arange(n_samples))
    boot_variance = np.array([variance(rows) for rows in boot])

    records: list[SensitivityResult] = []

    for i, name in enumerate(names):
        a_b = a.copy()
        a_b[:, i] = b[:, i]
        squared_diff = (y_a - evaluate(a_b)) ** 2
        s_t = squared_diff.mean() / (2 * total_variance)
        boot_s_t = squared_diff[boot].mean(axis=1) / (2 * boot_variance)
        low, high = np.percentile(boot_s_t, [2.5, 97.5])
        records.append(
            SensitivityResult(
                parameter=name,
                s_t=float(s_t),
                s_t_low=float(low),
                s_t_high=float(high),
                target_std=float(np.sqrt(total_variance)),
            )
        )

    return pd.DataFrame([r.model_dump() for r in records])


def compute_sensitivity(
    config: ConfigManager,
    target_name: str,
    n_samples: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    target_index = TARGET_NAMES.index(target_name)
    grid = config.ecm_parameters.model_dump()
    encoder = load(config.paths.sim_dir / "encode_model.pkl")
    model_root = config.paths.sim_dir / SimulationType.OPTIMIZATION.value

    frames = []

    for model_path in sorted(model_root.glob("*/*/surrogate_model.pkl")):
        building_type, city = model_path.parent.name, model_path.parent.parent.name
        surrogate_model = load(model_path)
        for code in SSP_ORDER:
            frame = total_order_indices(
                surrogate_model=surrogate_model,
                encoder=encoder,
                grid=grid,
                code=code,
                target_index=target_index,
                n_samples=n_samples,
                n_bootstrap=n_bootstrap,
                seed=config.optimization.seed,
            )
            frames.append(
                frame.assign(city=city, building_type=building_type, code=code)
            )
    columns = ["city", "building_type", "code", "parameter"]
    result = pd.concat(frames, ignore_index=True)
    return result[columns + [c for c in result.columns if c not in columns]]
