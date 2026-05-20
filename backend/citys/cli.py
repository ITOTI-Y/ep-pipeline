import asyncio
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger

from backend.citys._share import CitysFileName
from backend.utils.config import ConfigManager

app = typer.Typer()

config = ConfigManager()

CITYS_FILE_NAME = CitysFileName()


@app.command()
def download_epw() -> None:
    from backend.citys.io.epw import download_epw_dataset

    cfg = config
    asyncio.run(download_epw_dataset(Path(cfg.paths.epw_dir), cfg.citys.download))


@app.command()
def extract_epw() -> None:
    """Extract climate features from EPW files."""
    from backend.citys.core.feature import extract_all

    cfg = config
    output = Path(cfg.paths.citys_dir) / CITYS_FILE_NAME.epw_features
    extract_all(Path(cfg.paths.epw_dir), output)


@app.command()
def cluster_epw() -> None:
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
    output_dir = Path(cfg.paths.citys_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(output_dir / CITYS_FILE_NAME.epw_features)
    _corr, x, _feature_names, meta_df, prep_info = preprocess(df, cfg.citys.preprocess)

    with open(output_dir / CITYS_FILE_NAME.epw_features_process_info, "w") as f:
        json.dump(prep_info, f, indent=2)

    meta_df.to_csv(
        output_dir / CITYS_FILE_NAME.epw_meta_data, index=False, encoding="utf-8-sig"
    )

    z = compute_ward_linkage(x)
    metrics_df = evaluate_k_range(x, z, cfg.citys.cluster)
    metrics_df.to_csv(output_dir / CITYS_FILE_NAME.epw_k_metrics, index=False)

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
    )

    tmyx_files = {
        path.stem.split("_")[-1]: path for path in cfg.paths.epw_dir.glob("*.epw")
    }
    rep_rows = []
    for idx in qc_result.final_indices:
        row = df.iloc[idx].to_dict()
        row["file_path"] = tmyx_files[str(row["wmo_id"])].resolve()
        row["selection_type"] = qc_result.selection_types[idx]
        row["cluster_label"] = int(km_result.labels[idx])
        rep_rows.append(row)
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(
        output_dir / CITYS_FILE_NAME.epw_representative_cities,
        index=False,
        encoding="utf-8-sig",
    )

    all_assign = meta_df.copy()
    all_assign["cluster_label"] = km_result.labels
    all_assign["is_medoid"] = [i in km_result.medoid_indices for i in range(len(df))]
    all_assign["is_representative"] = [
        i in qc_result.final_indices for i in range(len(df))
    ]
    all_assign.to_csv(
        output_dir / CITYS_FILE_NAME.epw_cluster_assignments,
        index=False,
        encoding="utf-8-sig",
    )

    np.save(output_dir / CITYS_FILE_NAME.epw_ward_linkage, z)

    logger.info(f"Clustering complete: {len(qc_result.final_indices)} cities selected")


@app.command()
def download_dest() -> None:
    import json

    import pandas as pd

    from backend.citys.io.dest import download_dest_models, fetch_catalog

    cfg = config
    out = Path(cfg.paths.citys_dir)
    dest_dir = Path(cfg.paths.dest_dir)
    rep_df = pd.read_csv(out / CITYS_FILE_NAME.epw_representative_cities)
    _cities = rep_df["city"].unique().tolist()

    async def _run():
        original_catalog = await fetch_catalog()
        with open(out / CITYS_FILE_NAME.dest_catalog, "w") as f:
            json.dump([r.model_dump() for r in original_catalog], f, indent=4)
        downloaded_city = [c.name.split("_")[0] for c in dest_dir.glob(r"*.sqlite")] + [
            c.name.split("_")[0] for c in dest_dir.glob(r"*.accdb")
        ]
        catalog = [c for c in original_catalog if c.city not in downloaded_city]
        return await download_dest_models(catalog, cfg.citys.download, dest_dir)

    asyncio.run(_run())


@app.command()
def mapping_dest_to_tmyx() -> None:
    import pandas as pd

    from backend.citys.core.mapping import map_tmyx_to_dest

    cfg = config
    out = Path(cfg.paths.citys_dir)
    dest_dir = Path(cfg.paths.dest_dir)
    rep_df = pd.read_csv(out / CITYS_FILE_NAME.epw_representative_cities)
    meta_df = pd.read_csv(out / CITYS_FILE_NAME.epw_meta_data)
    labels = pd.read_csv(out / CITYS_FILE_NAME.epw_cluster_assignments)[
        "cluster_label"
    ].to_numpy()
    map_tmyx_to_dest(
        rep_df,
        labels,
        meta_df,
        dest_dir,
        out / CITYS_FILE_NAME.dest_coords,
        out / CITYS_FILE_NAME.dest_mapped_results,
    )


@app.command()
def plot() -> None:
    from backend.citys.viz.results import generation_all

    generation_all(config)


@app.command()
def run(
    epw: Annotated[bool, "--epw", typer.Option(help="Download EPW files")] = False,
    dest: Annotated[bool, "--dest", typer.Option(help="Download DeST models")] = False,
    plt: Annotated[bool, "--plt", "-p", typer.Option(help="Plot results")] = False,
    download: Annotated[
        bool, "--download", typer.Option(help="Download EPW and DeST files")
    ] = False,
    all: Annotated[bool, "--all", typer.Option(help="Run all steps")] = False,
) -> None:
    """Run complete pipeline: download -> extract -> cluster -> plot."""
    if all:
        epw = True
        dest = True
        plt = True
    if download:
        download_epw()
        download_dest()
    if epw:
        extract_epw()
        cluster_epw()
    if dest:
        mapping_dest_to_tmyx()
    if plt:
        plot()
    logger.info("City selection pipeline complete")
