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
def download_tmyx() -> None:
    from backend.citys.io.epw import download_tmyx_dataset

    cfg = config
    asyncio.run(
        download_tmyx_dataset(
            Path(cfg.paths.epw_dir), Path(cfg.paths.ddy_dir), cfg.citys.download
        )
    )


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
        build_energy_space,
        compute_ward_linkage,
        run_kmedoids,
        select_k_by_coverage,
    )
    from backend.citys.core.preprocess import preprocess
    from backend.citys.core.qc import run_qc

    cfg = config
    output_dir = Path(cfg.paths.citys_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(output_dir / CITYS_FILE_NAME.epw_features)
    _, _, _, meta_df, prep_info = preprocess(df, cfg.citys.preprocess)

    with open(output_dir / CITYS_FILE_NAME.epw_features_process_info, "w") as f:
        json.dump(prep_info, f, indent=2)

    meta_df.to_csv(
        output_dir / CITYS_FILE_NAME.epw_meta_data, index=False, encoding="utf-8-sig"
    )

    x_energy = build_energy_space(df, cfg.citys.preprocess.pca_variance)
    z = compute_ward_linkage(x_energy)
    if cfg.citys.cluster.override_k is not None:
        optimal_k = cfg.citys.cluster.override_k
        logger.info(f"Using override K={optimal_k}")

    else:
        optimal_k, coverage_df = select_k_by_coverage(x_energy, df, cfg.citys.cluster)
        coverage_df.to_csv(output_dir / CITYS_FILE_NAME.epw_k_metrics, index=False)
        logger.info(f"Coverage-selected K={optimal_k}")

    km_result = run_kmedoids(x_energy, optimal_k)

    forced_cities = cfg.citys.forced_cities

    qc_result = run_qc(
        km_result.medoid_indices,
        km_result.labels,
        x_energy,
        df,
        meta_df,
        forced_cities,
    )

    tmyx_files = {
        path.stem.split("_")[-1]: path for path in cfg.paths.epw_dir.glob("*.epw")
    }
    ddy_files = {
        path.stem.split("_")[-1]: path for path in cfg.paths.ddy_dir.glob("*.ddy")
    }
    rep_rows = []
    for idx in qc_result.final_indices:
        row = df.iloc[idx].to_dict()
        wmo_id = str(row["wmo_id"])
        row["epw_file_path"] = tmyx_files[wmo_id].resolve()
        row["ddy_file_path"] = ddy_files[wmo_id].resolve()
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
def mapping_dest_to_tmyx(
    epw_output_dir: Annotated[
        Path | None,
        typer.Option("-o", "--output-dir", help="Output directory for TMYX files"),
    ] = None,
) -> None:
    import pandas as pd

    from backend.citys.core.mapping import map_tmyx_to_dest

    cfg = config
    out = Path(cfg.paths.citys_dir)
    sqlite_dir = Path(cfg.paths.dest_dir) / "sqlite"
    rep_df = pd.read_csv(out / CITYS_FILE_NAME.epw_representative_cities)
    meta_df = pd.read_csv(out / CITYS_FILE_NAME.epw_meta_data)
    labels = pd.read_csv(out / CITYS_FILE_NAME.epw_cluster_assignments)[
        "cluster_label"
    ].to_numpy()
    result = map_tmyx_to_dest(
        rep_df,
        labels,
        meta_df,
        sqlite_dir,
        out / CITYS_FILE_NAME.dest_coords,
        out / CITYS_FILE_NAME.dest_mapped_results,
    )
    if epw_output_dir is not None:
        import shutil

        epw_output_dir.mkdir(parents=True, exist_ok=True)
        for _, row in result.iterrows():
            epw_file_paths = row["tmyx_epw_file_paths"]
            for file in epw_file_paths:
                path = Path(file)
                shutil.copy(path, epw_output_dir / path.name)


@app.command()
def plot() -> None:
    from backend.citys.viz.results import generation_all

    generation_all(config)


@app.command()
def run(
    tmyx: Annotated[bool, "--tmyx", typer.Option(help="Download TMYX files")] = False,
    dest: Annotated[bool, "--dest", typer.Option(help="Download DeST models")] = False,
    plt: Annotated[bool, "--plt", "-p", typer.Option(help="Plot results")] = False,
    download: Annotated[
        bool, "--download", typer.Option(help="Download EPW, DDY and DeST files")
    ] = False,
    all: Annotated[bool, "--all", typer.Option(help="Run all steps")] = False,
) -> None:
    """Run complete pipeline: download -> extract -> cluster -> plot."""
    if all:
        tmyx = True
        dest = True
        plt = True
    if tmyx:
        if download:
            download_tmyx()
        extract_epw()
        cluster_epw()
    if dest:
        if download:
            download_dest()
        mapping_dest_to_tmyx()
    if plt:
        plot()
    logger.info("City selection pipeline complete")
