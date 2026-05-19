from pathlib import Path

import cartopy.crs as ccrs
import pandas as pd
from cartopy.feature import ShapelyFeature

from backend.citys._share import VizFileName
from backend.citys.viz._share import (
    CENTER_LONGITUDE,
    GEO_PROVINCE_FILE,
    INSET_H,
    INSET_W,
    MAP_EXTENT,
    SCS_EXTENT,
    load_geojson,
)
from backend.viz.generator import ChartGenerator
from backend.viz.style import FigureWidth, JournalStyle

STYLE = JournalStyle()
VIZ_FILE_NAME = VizFileName()

_chart_generator = ChartGenerator()


def station_distribution(
    geo_dir: Path, tmy_df: pd.DataFrame, dest_df: pd.DataFrame
) -> None:
    fig, ax = _chart_generator.create_figure(
        width=FigureWidth.DOUBLE_COLUMN,
        aspect_ratio=0.6,
        proj="cyl",
        proj_kw={"lon_0": CENTER_LONGITUDE},
    )
    ax.format(
        title="TMYx Weather Stations and DeST Models",
        lonlim=(MAP_EXTENT[0], MAP_EXTENT[1]),
        latlim=(MAP_EXTENT[2], MAP_EXTENT[3]),
    )
    ax.scatter(
        tmy_df[~tmy_df["is_representative"]]["longitude"],
        tmy_df[~tmy_df["is_representative"]]["latitude"],
        transform=ccrs.PlateCarree(),
        s=3,
        c=STYLE.colors[0],
        zorder=10,
        marker=STYLE.markers[0],
        label="TMYx Weather Stations",
    )
    ax.scatter(
        tmy_df[tmy_df["is_representative"]]["longitude"],
        tmy_df[tmy_df["is_representative"]]["latitude"],
        transform=ccrs.PlateCarree(),
        s=8,
        c=STYLE.colors[1],
        edgecolor="black",
        linewidth=STYLE.line_width_thick / 3,
        zorder=15,
        marker=STYLE.markers[1],
        label="Representative TMYx Weather Stations",
    )
    ax.scatter(
        dest_df["longitude"],
        dest_df["latitude"],
        transform=ccrs.PlateCarree(),
        s=3,
        c=STYLE.colors[3],
        zorder=10,
        marker=STYLE.markers[3],
        label="DeST Models",
    )
    geoms = load_geojson(geo_dir / GEO_PROVINCE_FILE)
    ax.add_feature(
        ShapelyFeature(
            geoms,
            ccrs.PlateCarree(),
            **{
                "color": "grey",
                "edgecolor": "black",
                "linewidth": STYLE.line_width_thick / 3,
                "alpha": 0.2,
            },
        )
    )

    scs_ax = ax.inset_axes(
        [1 - INSET_W, 0.0, INSET_W, INSET_H],
        projection=ccrs.PlateCarree(central_longitude=CENTER_LONGITUDE),
    )
    scs_ax.format(
        lonlim=(SCS_EXTENT[0], SCS_EXTENT[1]),
        latlim=(SCS_EXTENT[2], SCS_EXTENT[3]),
    )
    geoms = load_geojson(geo_dir / GEO_PROVINCE_FILE)
    scs_ax.add_feature(
        ShapelyFeature(
            geoms,
            ccrs.PlateCarree(),
            **{
                "color": "grey",
                "edgecolor": "grey",
                "linewidth": STYLE.line_width_thick / 3,
                "alpha": 0.5,
            },
        )
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        fontsize=STYLE.font_size_small,
        ncol=1,
        frameon=False,
    )
    _chart_generator.save(fig, VIZ_FILE_NAME.station_distribution)
    pass
