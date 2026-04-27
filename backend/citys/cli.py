import asyncio
from pathlib import Path

import numpy as np
import typer
from loguru import logger

from backend.utils.config import ConfigManager

app = typer.Typer()

config = ConfigManager()


@app.command()
def download_epw() -> None:
    from backend.citys.io.epw import download_epw_dataset

    cfg = config
    asyncio.run(download_epw_dataset(Path(cfg.paths.epw_dir), cfg.citys.download))


@app.command()
def extract() -> None:
    """Extract climate features from EPW files."""
    from backend.citys.core.feature import extract_all

    cfg = config
    output = Path(cfg.paths.citys_dir) / "processed_features.csv"
    extract_all(Path(cfg.paths.epw_dir), output)


@app.command()
def cluster() -> None:
    import json

    import pandas as pd

    from backend.citys.core.cluster import (
        compute_ward_linkage,
        evaluate_k_range,
        run_kmedoids,
        select_optimal_k,
    )
    from backend.citys.core.preprocess import preprocess
    from backend.citys.core.qc import run_qc

    cfg = config
    out = Path(cfg.paths.citys_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out / "processed_features.csv")
    _corr, x, _feature_names, meta_df, prep_info = preprocess(df, cfg.citys.preprocess)

    with open(out / "output_preprocessing_info.json", "w") as f:
        json.dump(prep_info, f, indent=2)

    z = compute_ward_linkage(x)
    metrics_df = evaluate_k_range(x, z, cfg.citys.cluster)
    metrics_df.to_csv(out / "output_k_metrics.csv", index=False)

    if cfg.citys.cluster.override_k is not None:
        optimal_k = cfg.citys.cluster.override_k
        logger.info(f"Using override K={optimal_k}")
    else:
        optimal_k = select_optimal_k(metrics_df)
        logger.info(f"Auto-selected K={optimal_k}")

    km_result = run_kmedoids(x, optimal_k)

    forced_cities = cfg.citys.forced_cities

    qc_result = run_qc(
        km_result.medoid_indices,
        km_result.labels,
        x,
        df,
        meta_df,
        forced_cities,
        cfg.citys.qc,
    )

    rep_rows = []
    for idx in qc_result.final_indices:
        row = df.iloc[idx].to_dict()
        row["selection_type"] = qc_result.selection_types[idx]
        row["cluster_label"] = int(km_result.labels[idx])
        rep_rows.append(row)
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(
        out / "output_representative_cities.csv", index=False, encoding="utf-8-sig"
    )

    all_assign = meta_df.copy()
    all_assign["cluster_label"] = km_result.labels
    all_assign["is_medoid"] = [i in km_result.medoid_indices for i in range(len(df))]
    all_assign["is_representative"] = [
        i in qc_result.final_indices for i in range(len(df))
    ]
    all_assign.to_csv(
        out / "output_cluster_assignments.csv", index=False, encoding="utf-8-sig"
    )

    np.save(out / "cache_ward_linkage.npy", z)

    logger.info(f"Clustering complete: {len(qc_result.final_indices)} cities selected")


@app.command()
def download_dest() -> None:
    import pandas as pd

    from backend.citys.io.dest import download_dest_models, fetch_catalog

    cfg = config
    out = Path(cfg.paths.citys_dir)
    mapping_path = out / "output_dest_mapping.csv"
    if not mapping_path.exists():
        logger.error(f"{mapping_path} not found. Run 'city cluster' first.")
        raise typer.Exit(1)
    mapping = pd.read_csv(mapping_path)
    cities = mapping["dest_city"].unique().tolist()

    async def _run():
        catalog = await fetch_catalog(cfg.citys.download)
        return await download_dest_models(
            cities, catalog, cfg.citys.download, Path(cfg.paths.dest_dir)
        )

    asyncio.run(_run())


@app.command()
def plot() -> None:
    pass


@app.command()
def run_all() -> None:
    """Run complete pipeline: download -> extract -> cluster -> plot."""
    download_epw()
    extract()
    cluster()
    download_dest()
    plot()
    logger.info("City selection pipeline complete")
