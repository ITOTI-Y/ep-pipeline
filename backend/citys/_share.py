from typing import Final

from pydantic import BaseModel, ConfigDict

RANDOM_SEED: Final = 0


class CitysFileName(BaseModel):
    model_config = ConfigDict(
        frozen=True,
    )
    epw_features: str = "01_epw_features.csv"
    epw_features_process_info: str = "02_epw_features_process_info.json"
    epw_meta_data: str = "03_epw_meta_data.csv"
    epw_k_metrics: str = "04_epw_k_metrics.csv"
    epw_representative_cities: str = "05_epw_representative_cities.csv"
    epw_cluster_assignments: str = "06_epw_cluster_assignments.csv"
    epw_ward_linkage: str = "07_epw_ward_linkage.npy"
    dest_catalog: str = "08_dest_catalog.json"
    dest_coords: str = "09_dest_coords.csv"
    dest_mapped_results: str = "10_dest_mapped_results.json"


class VizFileName(BaseModel):
    model_config = ConfigDict(
        frozen=True,
    )
    station_distribution: str = "01_station_distribution.png"
    correlation_heatmap: str = "02_correlation_heatmap.png"
    pca_analysis: str = "03_pca_analysis.png"
    cluster_dendrogram: str = "04_cluster_dendrogram.png"
    k_metrics: str = "05_k_metrics.png"
