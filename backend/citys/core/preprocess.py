from typing import NamedTuple, TypedDict

import numpy as np
import pandas as pd
from citys.models.schemas import PreprocessConfigSchema
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from backend.citys.core._share import (
    GROUP_A_COLS,
    GROUP_B_COLS,
    GROUP_C_COLS,
    META_COLS,
)


class Info(TypedDict):
    n_pca_components: int
    pca_explained_variance_ratio: list[float]
    pca_cumulative_variance: float
    geo_weight: float
    final_dim: int


class PreprocessResult(NamedTuple):
    corr: NDArray[np.float64]
    x: NDArray[np.float64]
    feature_names: list[str]
    meta_df: pd.DataFrame
    info: Info


def preprocess(df: pd.DataFrame, cfg: PreprocessConfigSchema) -> PreprocessResult:
    meta_df = df[META_COLS].copy().reset_index(drop=True)

    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    scaler_c = StandardScaler()

    x_a = scaler_a.fit_transform(df[GROUP_A_COLS].to_numpy())
    x_b = scaler_b.fit_transform(df[GROUP_B_COLS].to_numpy())
    x_c = scaler_c.fit_transform(df[GROUP_C_COLS].to_numpy())

    corr = np.corrcoef(df[GROUP_A_COLS].to_numpy(), rowvar=False)

    pca = PCA(n_components=cfg.pca_variance, svd_solver="full")
    x_b_pca = pca.fit_transform(x_b)

    x_c_w = x_c * cfg.geo_weight
    x = np.hstack([x_a, x_b_pca, x_c_w])

    feature_names = (
        list(GROUP_A_COLS)
        + [f"PC{i + 1}" for i in range(x_b_pca.shape[1])]
        + [f"{c}_w" for c in GROUP_C_COLS]
    )

    info: Info = Info(
        n_pca_components=int(x_b_pca.shape[1]),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        pca_cumulative_variance=float(pca.explained_variance_ratio_.sum()),
        geo_weight=cfg.geo_weight,
        final_dim=x.shape[1],
    )

    return PreprocessResult(
        corr=corr,
        x=x,
        feature_names=feature_names,
        meta_df=meta_df,
        info=info,
    )
