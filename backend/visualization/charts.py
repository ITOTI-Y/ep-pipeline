from datetime import datetime
from pathlib import Path
from pickle import load
from typing import Any

import numpy as np
import pandas as pd
import ultraplot as uplt
from sqlalchemy import create_engine

from backend.models import BuildingType, SimulationResult
from backend.utils.config import ConfigManager
from backend.visualization.journal_style import (
    BUILDING_AND_ENVIRONMENT_STYLE,
    FigureWidth,
    ImageType,
    JournalStyle,
)
from backend.visualization.query import Query


def hour_of_year(dt: datetime) -> int:
    year_start = datetime(2017, 1, 1)
    delta = dt - year_start
    return int(delta.total_seconds() // 3600)


class ChartGenerator:
    """Chart generator for PV simulation results visualization."""

    def __init__(
        self,
        config: ConfigManager,
        style: JournalStyle | None = None,
    ):
        self.config = config
        self.paths = config.paths
        self.output_dir = config.paths.visualization_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.query = Query()
        self.style = style or BUILDING_AND_ENVIRONMENT_STYLE
        uplt.rc.update(self.style.get_rc_params())
        self.pv_results = self._pv_data_prepare()

    def _pv_data_prepare(self) -> list[SimulationResult]:
        pv_results = []
        pv_dir = self.paths.pv_dir
        for pv_file in pv_dir.glob("**/result.pkl"):
            with open(pv_file, "rb") as f:
                pv_data = load(f)
                pv_results.append(pv_data)
        return pv_results

    def create_figure(
        self,
        width: FigureWidth = FigureWidth.SINGLE_COLUMN,
        aspect_ratio: float = 0.3,
        ncols: int = 1,
        nrows: int = 1,
        **kwargs,
    ) -> tuple[uplt.Figure, Any]:
        """Create figure with journal-compliant dimensions.

        Args:
            width: Total figure width (not per-subplot).
            aspect_ratio: Height/width ratio for each subplot.
            ncols: Number of columns.
            nrows: Number of rows.
        """
        subplot_width = width.value / ncols
        subplot_height = subplot_width * aspect_ratio
        return uplt.subplots(
            ncols=ncols,
            nrows=nrows,
            refwidth=subplot_width,
            refheight=subplot_height,
            **kwargs,
        )

    def save(
        self,
        fig: uplt.Figure,
        building_type: BuildingType,
        name: str,
        image_type: ImageType = ImageType.LINE_ART,
        fmt: str | None = None,
    ) -> Path:
        """Save figure with journal-compliant format and DPI."""
        image_dir = self.output_dir / building_type.value
        image_dir.mkdir(parents=True, exist_ok=True)
        output_format = fmt or self.style.default_format
        path = image_dir / f"{name}.{output_format}"
        fig.save(path, dpi=image_type.value)
        return path

    def storage_soc(self, pv_result: SimulationResult) -> None:
        """Generate storage state of charge chart for a simulation result.

        Args:
            pv_result: Simulation result containing SQL path and building type.
        """
        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.35,
        )

        try:
            weather_code = (
                pv_result.sql_path.parent.name
                if pv_result.sql_path is not None
                else "Unknown"
            )
            building_type = pv_result.building_type
            capacity = self.config.storage.capacity[pv_result.building_type]

            engine = create_engine(f"sqlite:///{pv_result.sql_path}")  # type: ignore[arg-type]
            soc_data = pd.read_sql_query(self.query.STORAGE_SOC_QUERY, engine)
            soc_data["time"] = pd.to_datetime(  # type: ignore
                {
                    "year": 2017,
                    "month": soc_data["Month"].astype(int),
                    "day": soc_data["Day"].astype(int),
                    "hour": soc_data["Hour"].astype(int),
                }
            )
            soc_data["hour_of_year"] = soc_data["time"].apply(hour_of_year)
            soc_data["Value"] = round(
                soc_data["Value"] / 3600 / 1000 / capacity * 100, 2
            )
            groups = soc_data.groupby("KeyValue")
            for keyvalue, group in groups:
                hours = group["hour_of_year"].values
                soc = group["Value"].values
                ax.plot(
                    hours,
                    soc,
                    label="State of Charge",
                    linewidth=self.style.line_width,
                    alpha=0.8,
                    color=self.style.get_color(0),
                )
                ax.set_xlabel("Hour of Year")
                ax.set_ylabel("State of Charge (%)")
                ax.set_title(f"{weather_code} - {building_type}")

                ax.axhline(
                    y=100,
                    color=self.style.get_color(2),
                    linestyle="--",
                    linewidth=self.style.line_width_thick,
                    label="Full capacity (100%)",
                )
                ax.axhline(
                    y=50,
                    color="#888888",
                    linestyle=":",
                    linewidth=self.style.line_width,
                    alpha=0.5,
                    label="Half capacity (50%)",
                )

                ax.set_ylim(0, 110)
                ax.set_yticks(np.arange(0, 110, 20))
                ax.set_xlim(0, 8760)
                ax.set_xticks(np.arange(0, 8760, 1000))

                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0, -0.15),
                    ncol=3,
                    frameon=False,
                )

                ax.annotate(
                    f"Capacity: {capacity} kWh",
                    xy=(0.02, 0.98),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                    fontsize=self.style.font_size_small,
                )

                self.save(
                    fig,
                    building_type,
                    f"F01-{weather_code}-{building_type}-{keyvalue}-Storage SOC",
                )
        finally:
            uplt.close(fig)

    def typical_day_storage_soc(self, pv_result: SimulationResult) -> None:
        """Generate typical day storage operation comparison chart.

        Compares summer (July 12-18) and winter (January 12-18) typical days,
        showing demand, PV generation, and storage charge/discharge patterns.

        Args:
            pv_result: Simulation result containing SQL path and building type.
        """
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.618,
            ncols=2,
            sharey=True,
        )
        titles = ["Summer (July 12-18)", "Winter Day (January 12-18)"]
        queries = [
            self.query.TYPICAL_SUMMER_DAY_SOC_QUERY,
            self.query.TYPICAL_WINTER_DAY_SOC_QUERY,
        ]

        def _plot_single_day(
            ax: uplt.Axes,
            hours: np.ndarray,
            demand: np.ndarray,
            pv: np.ndarray,
            storage_charge: np.ndarray,
            storage_discharge: np.ndarray,
        ):
            ax.plot(
                hours,
                demand,
                linewidth=self.style.line_width_thick,
                color=self.style.get_color(0),
                label="Demand",
            )
            ax.fill_between(
                hours,
                pv,
                alpha=0.6,
                color=self.style.get_color(4),
                label="PV Generation",
            )
            ax.bar(
                hours,
                storage_charge,
                width=0.6,
                alpha=0.7,
                color=self.style.get_color(3),
                label="Storage Charge",
            )
            ax.bar(
                hours,
                storage_discharge,
                width=0.6,
                alpha=0.7,
                color=self.style.get_color(2),
                label="Storage Discharge",
            )
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Power (kW)")

        try:
            weather_code = (
                pv_result.sql_path.parent.name
                if pv_result.sql_path is not None
                else "Unknown"
            )
            building_type = pv_result.building_type
            axs.format(
                abc="a.",
                abcloc="ul",
                suptitle=f"{weather_code} - {building_type} - Typical Day Operation",
            )
            axs[0].legend(
                loc="upper left", bbox_to_anchor=(0, -0.15), ncol=4, frameon=False
            )

            engine = create_engine(f"sqlite:///{pv_result.sql_path}")  # type: ignore[arg-type]
            for ax, q, title in zip(axs, queries, titles, strict=True):
                typical_df = pd.read_sql_query(q, engine)
                _plot_single_day(
                    ax,
                    typical_df.index.values.astype(int),
                    typical_df["demand_value"].values / 1000,
                    typical_df["pv_value"].values / 1000,
                    typical_df["storage_charge_value"].values / 1000,
                    typical_df["storage_discharge_value"].values / 1000,
                )
                ax.set_title(title)
            self.save(
                fig,
                building_type,
                f"F02-{weather_code}-{building_type}-Typical Day Storage SOC",
            )
        finally:
            uplt.close(fig)

    def generate_all(self) -> None:
        for pv_result in self.pv_results:
            self.storage_soc(pv_result)
            self.typical_day_storage_soc(pv_result)
