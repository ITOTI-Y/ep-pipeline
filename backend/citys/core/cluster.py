from typing import NamedTuple

import numpy as np
import pandas as pd
from kmedoids import fasterpam
from loguru import logger
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from backend.citys._share import RANDOM_SEED
from backend.citys.models.schemas import ClusterConfigSchema


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


def select_optimal_k(metrics_df: pd.DataFrame) -> int:
    k_sil = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    k_ch = int(metrics_df.loc[metrics_df["calinski_harabasz"].idxmax(), "k"])

    k_gap = None
    for i in range(len(metrics_df) - 1):
        gap_k = metrics_df.iloc[i]["gap_statistic"]
        gap_k1 = metrics_df.iloc[i + 1]["gap_statistic"]
        sk1 = metrics_df.iloc[i + 1]["gap_sk"]
        if gap_k >= gap_k1 - sk1:
            k_gap = int(metrics_df.iloc[i]["k"])
            break
    if k_gap is None:
        k_gap = int(metrics_df.loc[metrics_df["gap_statistic"].idxmax(), "k"])

    candidates = [k_sil, k_ch, k_gap]
    logger.info(f"K votes: Sil={k_sil}, CH={k_ch}, Gap={k_gap}")
    for k in candidates:
        if candidates.count(k) >= 2:
            return k
    return k_sil


def run_kmedoids(x: NDArray, k: int) -> KMedoidsResult:
    dist_matrix = squareform(pdist(x, metric="euclidean"))
    result = fasterpam(dist_matrix, k, random_state=RANDOM_SEED)
    labels = np.array(result.labels)
    medoid_indices = np.array(result.medoids)
    logger.info(f"K-Medoids: k={k}, loss={result.loss:.2f}")
    return KMedoidsResult(labels, medoid_indices, result.loss)
