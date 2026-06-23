from typing import Final, NamedTuple

import numpy as np
import pandas as pd
from kmedoids import fasterpam
from loguru import logger
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from backend.citys._share import RANDOM_SEED
from backend.citys.core._share import GROUP_A_COLS, GROUP_B_COLS
from backend.citys.models.schemas import ClusterConfigSchema

COVERAGE_GROUPS: Final = {
    "temp": ["hdd18", "cdd18"],
    "solar": ["annual_ghi", "annual_dhi"],
    "wind": ["annual_mean_wind_speed"],
}


def build_energy_space(df: pd.DataFrame, pca_variance: float) -> NDArray:
    x_a = StandardScaler().fit_transform(df[GROUP_A_COLS].to_numpy())
    x_b = StandardScaler().fit_transform(df[GROUP_B_COLS].to_numpy())
    x_b_pca = PCA(n_components=pca_variance, svd_solver="full").fit_transform(x_b)
    x_b_pca = StandardScaler().fit_transform(x_b_pca)
    return np.hstack([x_a, x_b_pca])


def select_k_by_coverage(
    x_energy: NDArray, df: pd.DataFrame, cfg: ClusterConfigSchema
) -> tuple[int, pd.DataFrame]:
    resources = {
        name: StandardScaler().fit_transform(df[cols].to_numpy())
        for name, cols in COVERAGE_GROUPS.items()
    }
    tol = {
        "temp": cfg.coverage_tol.temp,
        "solar": cfg.coverage_tol.solar,
        "wind": cfg.coverage_tol.wind,
    }
    dist = squareform(pdist(x_energy, metric="euclidean"))
    rows = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        result = fasterpam(dist, k, random_state=RANDOM_SEED)
        assigned = np.asarray(result.medoids)[np.asarray(result.labels)]
        record: dict[str, float | int | bool] = {"k": k}
        meets = True
        for name, mat in resources.items():
            p95 = float(np.percentile(np.linalg.norm(mat - mat[assigned], axis=1), 95))
            record[f"p95_{name}"] = p95
            meets = meets and p95 <= tol[name]
        record["meets_all"] = meets
        rows.append(record)
    coverage_df = pd.DataFrame(rows)
    feasible = coverage_df.loc[coverage_df["meets_all"], "k"]
    if len(feasible):
        k = int(feasible.min())
        logger.info(f"Coverage K={k} (per-resource P95 within {tol})")
    else:
        k = int(cfg.k_max)
        logger.warning(
            f"No K in [{cfg.k_min}, {cfg.k_max}] meets per-resource tol {tol}; "
            f"using k_max={k}. Relax wind tolerance or raise k_max."
        )
    return k, coverage_df


class KMedoidsResult(NamedTuple):
    labels: NDArray[np.intp]
    medoid_indices: NDArray[np.intp]
    loss: float


def compute_ward_linkage(x: NDArray) -> NDArray:
    return linkage(x, method="ward", metric="euclidean")


def evaluate_k_range(x: NDArray, z: NDArray, cfg: ClusterConfigSchema) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_SEED)
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    results = []

    for k in range(cfg.k_min, cfg.k_max + 1):
        labels = fcluster(z, t=k, criterion="maxclust")
        sil = silhouette_score(x, labels)
        ch = calinski_harabasz_score(x, labels)

        wk = max(_compute_wk(x, labels), 1e-300)
        log_wk_refs = []
        for _ in range(cfg.n_gap_refs):
            x_ref = rng.uniform(x_min, x_max, size=x.shape)
            z_ref = linkage(x_ref, method="ward")
            labels_ref = fcluster(z_ref, t=k, criterion="maxclust")
            log_wk_refs.append(np.log(max(_compute_wk(x_ref, labels_ref), 1e-300)))
        log_wk_refs_arr = np.array(log_wk_refs)
        gap = float(np.mean(log_wk_refs_arr) - np.log(wk))
        gap_sk = float(np.std(log_wk_refs_arr) * np.sqrt(1 + 1 / cfg.n_gap_refs))

        results.append(
            {
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "gap_statistic": gap,
                "gap_sk": gap_sk,
            }
        )

    return pd.DataFrame(results)


def _compute_wk(x: NDArray, labels: NDArray) -> float:
    wk = 0.0
    for label in np.unique(labels):
        cluster = x[labels == label]
        center = cluster.mean(axis=0)
        wk += float(np.sum((cluster - center) ** 2))
    return wk


def select_optimal_k(metrics_df: pd.DataFrame, sil_tol: float = 0.1) -> int:
    sil_max = float(metrics_df["silhouette"].max())
    threshold = sil_max * (1.0 - sil_tol)
    eligible = metrics_df.loc[metrics_df["silhouette"] >= threshold, "k"]
    k = int(eligible.min())
    logger.info(
        f"Selected K={k} (silhouette within {sil_tol:.0%} of peak {sil_max:.4f})"
    )
    return k


def run_kmedoids(x: NDArray, k: int) -> KMedoidsResult:
    dist_matrix = squareform(pdist(x, metric="euclidean"))
    result = fasterpam(dist_matrix, k, random_state=RANDOM_SEED)
    labels = np.array(result.labels)
    medoid_indices = np.array(result.medoids)
    logger.info(f"K-Medoids: k={k}, loss={result.loss:.2f}")
    return KMedoidsResult(labels, medoid_indices, result.loss)
