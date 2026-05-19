from pathlib import Path

import cartopy.crs as ccrs
import pandas as pd
from cartopy.feature import ShapelyFeature

from backend.citys._share import CitysFileName, VizFileName
from backend.citys.core._share import COL_DISPLAY, GROUP_A_COLS, GROUP_C_COLS
from backend.citys.viz._share import (
    CENTER_LONGITUDE,
    GEO_PROVINCE_FILE,
    INSET_H,
    INSET_W,
    MAP_EXTENT,
    SCS_EXTENT,
    load_geojson,
)
from backend.utils.config import ConfigManager
from backend.viz.generator import ChartGenerator
from backend.viz.style import FigureWidth, JournalStyle

STYLE = JournalStyle()
VIZ_FILE_NAME = VizFileName()
CITYS_FILE_NAME = CitysFileName()

_chart_generator = ChartGenerator()


def station_distribution(
    geo_dir: Path, tmy_df: pd.DataFrame, dest_df: pd.DataFrame
) -> None:
    fig, ax = _chart_generator.create_figure(
        width=FigureWidth.DOUBLE_COLUMN,
        aspect_ratio=0.62,
        proj="cyl",
        proj_kw={"lon_0": CENTER_LONGITUDE},
    )
    ax.format(
        lonlim=(MAP_EXTENT[0], MAP_EXTENT[1]),
        latlim=(MAP_EXTENT[2], MAP_EXTENT[3]),
        lonlocator=10,
        latlocator=10,
        lonlabels="b",
        latlabels="l",
        grid=True,
        gridcolor="black",
        gridlinewidth=STYLE.line_width_thick / 2,
        gridlinestyle=":",
        gridminor=False,
        labelsize=STYLE.font_size_small,
        ticklabelsize=STYLE.font_size_small,
    )
    ax.set_facecolor("#eaf2f8")
    geoms = load_geojson(geo_dir / GEO_PROVINCE_FILE)
    ax.add_feature(
        ShapelyFeature(
            geoms,
            ccrs.PlateCarree(),
            facecolor="#f2f2f2",
            edgecolor="grey",
            linewidth=STYLE.line_width_thick / 3,
            zorder=1,
        )
    )

    common_kw = {
        "transform": ccrs.PlateCarree(),
        "edgecolor": "white",
        "linewidth": 0.35,
    }
    n_regular = (~tmy_df["is_representative"]).sum()
    n_repr = tmy_df["is_representative"].sum()
    n_dest = len(dest_df)

    ax.scatter(
        tmy_df.loc[~tmy_df["is_representative"], "longitude"],
        tmy_df.loc[~tmy_df["is_representative"], "latitude"],
        s=10,
        c=STYLE.colors[6],
        marker=STYLE.markers[0],
        alpha=0.85,
        zorder=10,
        label=f"Weather stations (n={n_regular})",
        **common_kw,
    )
    ax.scatter(
        dest_df["longitude"],
        dest_df["latitude"],
        s=12,
        c=STYLE.colors[3],
        marker=STYLE.markers[3],
        alpha=0.85,
        zorder=11,
        label=f"DeST models (n={n_dest})",
        **common_kw,
    )
    ax.scatter(
        tmy_df.loc[tmy_df["is_representative"], "longitude"],
        tmy_df.loc[tmy_df["is_representative"], "latitude"],
        s=24,
        c=STYLE.colors[1],
        marker=STYLE.markers[1],
        linewidth=0.5,
        zorder=15,
        label=f"Representative stations (n={n_repr})",
        transform=ccrs.PlateCarree(),
        edgecolor="white",
    )

    scs_ax = ax.inset_axes(
        [1 - INSET_W - 0.005, 0.005, INSET_W, INSET_H],
        projection=ccrs.PlateCarree(central_longitude=CENTER_LONGITUDE),
    )
    scs_ax.format(
        lonlim=(SCS_EXTENT[0], SCS_EXTENT[1]),
        latlim=(SCS_EXTENT[2], SCS_EXTENT[3]),
        grid=False,
    )
    scs_ax.set_facecolor("#eaf2f8")
    geoms = load_geojson(geo_dir / GEO_PROVINCE_FILE)
    scs_ax.add_feature(
        ShapelyFeature(
            geoms,
            ccrs.PlateCarree(),
            facecolor="#f2f2f2",
            edgecolor="grey",
            linewidth=0.3,
            zorder=1,
        )
    )
    ax.legend(
        loc="lower left",
        fontsize=STYLE.font_size_small,
        ncol=1,
        frameon=False,
    )
    _chart_generator.save(fig, VIZ_FILE_NAME.station_distribution)


def correlation_heatmap(df: pd.DataFrame) -> None:
    import numpy as np
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    fig, ax = _chart_generator.create_figure(
        width=FigureWidth.ONE_HALF_COLUMN,
        aspect_ratio=1.0,
    )
    cols = GROUP_A_COLS + GROUP_C_COLS
    corr = df[cols].corr()

    order = leaves_list(
        linkage(squareform(1 - corr.abs(), checks=False), method="average")
    )
    cols = [cols[i] for i in order]
    corr = corr.iloc[order, order]

    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    corr_masked = corr.mask(mask)

    m = ax.heatmap(
        corr_masked,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        labels=True,
        precision=2,
        edgecolor="white",
        linewidth=0.6,
        labels_kw={"size": STYLE.font_size_small},
    )
    ax.colorbar(
        m,
        loc="r",
        length=1.0,
        label=r"Pearson correlation $r$",
        ticks=[-1, -0.5, 0, 0.5, 1],
    )
    display_labels = [COL_DISPLAY[c] for c in cols]
    ax.format(
        xticks=range(len(cols)),
        xticklabels=display_labels,
        yticks=range(len(cols)),
        yticklabels=display_labels,
    )
    _chart_generator.save(fig, VIZ_FILE_NAME.correlation_heatmap)
    pass


def generation_all(config: ConfigManager):
    import pandas as pd

    epw_cluster_df = pd.read_csv(
        Path(config.paths.citys_dir) / CITYS_FILE_NAME.epw_cluster_assignments
    )
    epw_feature_df = pd.read_csv(
        Path(config.paths.citys_dir) / CITYS_FILE_NAME.epw_features
    )
    dest_df = pd.read_csv(Path(config.paths.citys_dir) / CITYS_FILE_NAME.dest_coords)

    station_distribution(config.paths.geo_dir, epw_cluster_df, dest_df)
    # correlation_heatmap(epw_feature_df)
    pass
