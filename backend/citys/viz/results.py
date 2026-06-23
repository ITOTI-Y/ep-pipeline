from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from cartopy.feature import ShapelyFeature

from backend.citys._share import CitysFileName, VizFileName
from backend.citys.core._share import (
    COL_DISPLAY,
    GROUP_A_COLS,
    GROUP_B_COLS,
    GROUP_C_COLS,
)
from backend.citys.models.schemas import CitySelectionConfigSchema
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


def pca_analysis(df: pd.DataFrame) -> None:

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x_b = StandardScaler().fit_transform(df[GROUP_B_COLS].to_numpy())
    pca = PCA().fit(x_b)
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)

    fig, axs = _chart_generator.create_figure(
        width=FigureWidth.ONE_HALF_COLUMN,
        aspect_ratio=0.5,
        ncols=1,
        nrows=1,
        sharex=False,
        sharey=False,
    )

    ax1 = axs[0]
    ax1.bar(range(len(var)), var, color=STYLE.colors[0], label="Individual")
    ax1.plot(
        range(len(var)),
        cum,
        "o-",
        ms=3,
        lw=STYLE.line_width,
        color=STYLE.colors[1],
        label="Cumulative",
    )
    ax1.axhline(
        y=0.95,
        color=STYLE.colors[2],
        linestyle="--",
        linewidth=STYLE.line_width,
        label="95%",
    )
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    ax1.axvline(
        x=n95,
        color="grey",
        linestyle=":",
        linewidth=STYLE.line_width * 0.6,
        zorder=1,
    )
    ax1.text(
        n95 + 0.4,
        0.05,
        f"n = {n95}",
        fontsize=STYLE.font_size_small,
        color="grey",
    )
    ax1.format(
        xlabel="Principal component",
        ylabel="Variance ratio",
    )
    ax1.legend(loc="lr", ncol=1, frameon=False)

    _chart_generator.save(fig, VIZ_FILE_NAME.pca_analysis)


def cluster_dendrogram(z: np.ndarray, k: int) -> None:
    from scipy.cluster.hierarchy import dendrogram

    n = z.shape[0] + 1
    cut_dist = (z[n - k - 1, 2] + z[n - k, 2]) / 2

    fig, ax = _chart_generator.create_figure(
        width=FigureWidth.DOUBLE_COLUMN,
        aspect_ratio=0.5,
    )
    dendrogram(
        z, truncate_mode="lastp", p=60, color_threshold=cut_dist, ax=ax, no_labels=True
    )
    ax.axhline(
        cut_dist, color=STYLE.colors[1], linestyle="--", linewidth=STYLE.line_width
    )
    ax.format(
        xlabel="Cluster", ylabel="Ward Distance", title=f"Ward Dendrogram (K={k})"
    )
    _chart_generator.save(fig, VIZ_FILE_NAME.cluster_dendrogram)


def k_metrics(feature_df: pd.DataFrame, cfg: CitySelectionConfigSchema, k: int) -> None:
    from kmedoids import fasterpam
    from scipy.spatial.distance import pdist, squareform

    from backend.citys._share import RANDOM_SEED
    from backend.citys.core.cluster import build_energy_space

    resources = {
        "hdd18": r"$\mathrm{HDD}_{18}$",
        "cdd18": r"$\mathrm{CDD}_{18}$",
        "annual_ghi": "GHI",
        "annual_mean_wind_speed": "Wind speed",
    }
    x_energy = build_energy_space(feature_df, cfg.preprocess.pca_variance)
    dist = squareform(pdist(x_energy, metric="euclidean"))
    ranges = {c: float(feature_df[c].max() - feature_df[c].min()) for c in resources}
    ks = list(range(cfg.cluster.k_min, cfg.cluster.k_max + 1))
    curves: dict[str, list[float]] = {c: [] for c in resources}
    for kk in ks:
        result = fasterpam(dist, kk, random_state=RANDOM_SEED)
        assigned = np.asarray(result.medoids)[np.asarray(result.labels)]
        for c in resources:
            v = feature_df[c].to_numpy()
            p95 = float(np.percentile(np.abs(v - v[assigned]), 95))
            curves[c].append(p95 / ranges[c] * 100.0)

    fig, axs = _chart_generator.create_figure(
        width=FigureWidth.DOUBLE_COLUMN, aspect_ratio=0.6, ncols=1
    )
    ax = axs[0]
    ax.axhspan(10.0, 15.0, color="gray", alpha=0.12, label="Target 10-15%")
    for i, (col, label) in enumerate(resources.items()):
        ax.plot(
            ks,
            curves[col],
            marker=STYLE.markers[i % len(STYLE.markers)],
            ms=3,
            lw=STYLE.line_width,
            color=STYLE.colors[i % len(STYLE.colors)],
            label=label,
        )
    ax.axvline(k, color="red", linestyle="--", linewidth=STYLE.line_width)
    ax.legend(loc="ur", ncol=1, frameon=False)
    ax.format(xlabel="K", ylabel="P95 error (% of national range)")
    _chart_generator.save(fig, VIZ_FILE_NAME.k_metrics)


def generation_all(config: ConfigManager):
    import pandas as pd

    epw_cluster_df = pd.read_csv(
        Path(config.paths.citys_dir) / CITYS_FILE_NAME.epw_cluster_assignments
    )
    epw_feature_df = pd.read_csv(
        Path(config.paths.citys_dir) / CITYS_FILE_NAME.epw_features
    )
    dest_df = pd.read_csv(Path(config.paths.citys_dir) / CITYS_FILE_NAME.dest_coords)
    ward_linkage = np.load(
        Path(config.paths.citys_dir) / CITYS_FILE_NAME.epw_ward_linkage
    )

    station_distribution(config.paths.geo_dir, epw_cluster_df, dest_df)
    correlation_heatmap(epw_feature_df)
    pca_analysis(epw_feature_df)
    cluster_dendrogram(ward_linkage, epw_cluster_df["cluster_label"].nunique())
    k_metrics(
        epw_feature_df,
        config.citys,
        epw_cluster_df["cluster_label"].nunique(),
    )
