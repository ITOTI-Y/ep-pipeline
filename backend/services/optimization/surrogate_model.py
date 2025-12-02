from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # type: ignore[possibly-missing-import]

from backend.utils.config import ConfigManager


class ISurrogateModel(ABC):
    @abstractmethod
    def __init__(self, config: ConfigManager) -> None:
        pass

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> None:
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
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            multi_strategy="multi_output_tree",
        )
        self._x_test: np.ndarray = np.array([])
        self._y_test: np.ndarray = np.array([])

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self._seed
        )
        self._model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        self._x_test = x_test  # type: ignore[assignment]
        self._y_test = y_test  # type: ignore[assignment]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def evaluate(self) -> None:
        pass
