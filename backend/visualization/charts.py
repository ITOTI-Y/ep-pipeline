import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pickle import load
from typing import Any

import numpy as np
import pandas as pd
import ultraplot as uplt
from sqlalchemy import create_engine

from backend.models import SimulationResult
from backend.utils.config import ConfigManager
from backend.visualization.journal_style import (
    BUILDING_AND_ENVIRONMENT_STYLE,
    FigureWidth,
    ImageType,
    JournalStyle,
)
from backend.visualization.query import Query

WEATHER_ORDER = ["TMY", "SSP126", "SSP245", "SSP370", "SSP434", "SSP585"]
BUILDING_ORDER = [
    "OfficeLarge",
    "OfficeMedium",
    "MultiFamilyResidential",
    "SingleFamilyResidential",
    "ApartmentHighRise",
]

BUILDING_NAME = {
    "OfficeLarge": "Large Office",
    "OfficeMedium": "Medium Office",
    "MultiFamilyResidential": "Multi-Family",
    "SingleFamilyResidential": "Single-Family",
    "ApartmentHighRise": "High-Rise Apt.",
}

style = JournalStyle()

BUILDING_COLOR = {
    "OfficeLarge": style.get_color(0),
    "OfficeMedium": style.get_color(1),
    "MultiFamilyResidential": style.get_color(2),
    "SingleFamilyResidential": style.get_color(3),
    "ApartmentHighRise": style.get_color(4),
}

GAS_BUILDING_TYPE = ["OfficeLarge", "OfficeMedium", "ApartmentHighRise"]


@dataclass
class Prefix:
    baseline: str = "B"
    pv: str = "P"
    optimization: str = "O"


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
        self.csv_dir = config.paths.csv_dir
        self.output_dir = config.paths.visualization_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.query = Query()
        self.style = style or BUILDING_AND_ENVIRONMENT_STYLE
        uplt.rc.update(self.style.get_rc_params())
        self.pv_results: dict[str, dict[str, SimulationResult]] = defaultdict(dict)
        self._pv_data_prepare()
        self.baseline_results: dict[str, dict[str, SimulationResult]] = defaultdict(
            dict
        )
        self._baseline_data_prepare()
        self.optimization_results: dict[str, dict[str, SimulationResult]] = defaultdict(
            dict
        )
        self._optimization_data_prepare()

    def _pv_data_prepare(self) -> None:
        pv_dir = self.paths.pv_dir
        for pv_file in pv_dir.glob("**/result.pkl"):
            with open(pv_file, "rb") as f:
                pv_data = load(f)
                building_type = pv_data.building_type
                weather_code = pv_data.sql_path.parent.name
                self.pv_results[building_type][weather_code] = pv_data

    def _baseline_data_prepare(self) -> None:
        baseline_dir = self.paths.baseline_dir
        for baseline_file in baseline_dir.glob("**/result.pkl"):
            with open(baseline_file, "rb") as f:
                baseline_data = load(f)
                building_type = baseline_data.building_type
                weather_code = baseline_data.sql_path.parent.name
                self.baseline_results[building_type][weather_code] = baseline_data

    def _optimization_data_prepare(self) -> None:
        optimization_dir = self.paths.optimization_dir
        for optimization_file in optimization_dir.glob("**/result.pkl"):
            with open(optimization_file, "rb") as f:
                optimization_data = load(f)
                building_type = optimization_data.building_type
                weather_code = optimization_data.sql_path.parent.name
                self.optimization_results[building_type][weather_code] = (
                    optimization_data
                )

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
        name: str,
        building_type: str | None = None,
        image_type: ImageType = ImageType.LINE_ART,
        fmt: str | None = None,
    ) -> Path:
        """Save figure with journal-compliant format and DPI."""
        image_dir = self.output_dir / (building_type if building_type else "")
        image_dir.mkdir(parents=True, exist_ok=True)
        output_format = fmt or self.style.default_format
        path = image_dir / f"{name}.{output_format}"
        fig.save(path, dpi=image_type.value)
        return path

    def storage_soc(self) -> None:
        """Generate storage state of charge chart for a simulation result.

        Args:
            pv_result: Simulation result containing SQL path and building type.
        """
        for building_type in BUILDING_ORDER:
            fig, axs = self.create_figure(
                width=FigureWidth.DOUBLE_COLUMN,
                aspect_ratio=0.35,
                ncols=1,
                nrows=len(WEATHER_ORDER),
            )
            axs.format(
                abc="a",
                abcloc="ul",
                suptitle=f"{building_type} - Storage State of Charge",
            )
            axs[-1].legend(
                loc="upper left",
                bbox_to_anchor=(0, -0.15),
                ncol=3,
                frameon=False,
            )
            for i, weather_code in enumerate(WEATHER_ORDER):
                pv_result = self.pv_results[building_type][weather_code]
                capacity = self.config.storage.capacity[building_type]
                engine = create_engine(f"sqlite:///{pv_result.sql_path}")
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
                hours = soc_data["hour_of_year"].values
                soc = soc_data["Value"].values

                axs[i].plot(
                    hours,
                    soc,
                    label="State of Charge",
                    linewidth=self.style.line_width,
                    alpha=0.8,
                    color=self.style.get_color(0),
                )

                axs[i].set_xlabel("Hour of Year")
                axs[i].set_ylabel("State of Charge (%)")
                axs[i].set_title(f"{weather_code} - {BUILDING_NAME[building_type]}")

                axs[i].axhline(
                    y=100,
                    color=self.style.get_color(2),
                    linestyle="--",
                    linewidth=self.style.line_width_thick,
                    label="Full capacity (100%)",
                )
                axs[i].axhline(
                    y=50,
                    color="#888888",
                    linestyle=":",
                    linewidth=self.style.line_width,
                    alpha=0.5,
                    label="Half capacity (50%)",
                )

                axs[i].set_ylim(0, 110)
                axs[i].set_yticks(np.arange(0, 110, 20))
                axs[i].set_xlim(0, 8760)
                axs[i].set_xticks(np.arange(0, 8760, 1000))

                axs[i].annotate(
                    f"Capacity: {capacity} kWh",
                    xy=(0.98, 0.98),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=self.style.font_size_small,
                )
            self.save(
                fig,
                f"{Prefix.pv}-{building_type}-Storage SOC",
                building_type,
            )

    def typical_day_storage_soc(self) -> None:
        """Generate typical day storage operation comparison chart.

        Compares summer (July 12-18) and winter (January 12-18) typical days,
        showing demand, PV generation, and storage charge/discharge patterns.

        Args:
            pv_result: Simulation result containing SQL path and building type.
        """
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

        for building_type in BUILDING_ORDER:
            fig, axs = self.create_figure(
                width=FigureWidth.DOUBLE_COLUMN,
                aspect_ratio=0.618,
                ncols=2,
                nrows=6,
            )
            axs.format(
                abc="a",
                abcloc="ul",
            )
            axs[-1].legend(
                loc="upper left",
                bbox_to_anchor=(0, -0.15),
                ncol=4,
                frameon=False,
            )
            for i, weather_code in enumerate(WEATHER_ORDER):
                pv_result = self.pv_results[building_type][weather_code]
                engine = create_engine(f"sqlite:///{pv_result.sql_path}")
                summer_df = pd.read_sql_query(queries[0], engine)
                _plot_single_day(
                    axs[i * 2],
                    summer_df.index.values.astype(int),
                    summer_df["demand_value"].values / 1000,
                    summer_df["pv_value"].values / 1000,
                    summer_df["storage_charge_value"].values / 1000,
                    summer_df["storage_discharge_value"].values / 1000,
                )
                axs[i * 2].set_title(titles[0] + f" - {weather_code}")

                winter_df = pd.read_sql_query(queries[1], engine)
                _plot_single_day(
                    axs[i * 2 + 1],
                    winter_df.index.values.astype(int),
                    winter_df["demand_value"].values / 1000,
                    winter_df["pv_value"].values / 1000,
                    winter_df["storage_charge_value"].values / 1000,
                    winter_df["storage_discharge_value"].values / 1000,
                )
                axs[i * 2 + 1].set_title(titles[1] + f" - {weather_code}")

            self.save(
                fig,
                f"{Prefix.pv}-{building_type}-Typical Day Storage SOC",
                building_type,
            )

    def baseline_result(self) -> None:
        """Generate baseline result chart.

        Args:
            baseline_result: Baseline result containing SQL path and building type.
        """
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.15,
            ncols=1,
            nrows=5,
        )
        axs.format(
            abc="a",
            abcloc="ul",
            suptitle="Baseline EUI Comparison",
        )
        data = self.baseline_results.copy()

        for i, (building_type, weather_codes) in enumerate(data.items()):
            eui = {}
            for weather_code, baseline_result in weather_codes.items():
                eui[weather_code] = baseline_result.total_source_eui
            sorted_eui = {k: eui[k] for k in WEATHER_ORDER if k in eui}
            labels = list(sorted_eui.keys())
            values = list(sorted_eui.values())
            axs[i].bar(
                labels,
                values,
                color=self.style.get_color(i),
                bar_labels=True,
                bar_labels_kw={"fmt": "%.1f"},
            )
            y_min = math.floor(min(values) / 100) * 100
            y_max = math.ceil(max(values) / 100) * 100
            axs[i].set_ylim(y_min, y_max)
            axs[i].set_yticks(np.arange(y_min, y_max + 1, 25))
            axs[i].set_title(BUILDING_NAME[building_type])
            axs[i].set_xlabel("Weather Code")
            axs[i].set_ylabel("EUI (kWh/m²/yr)")

        self.save(
            fig,
            f"{Prefix.baseline}-Baseline EUI Comparison",
            building_type=None,
        )

    def optimal_improvement_comparison(self) -> None:
        """Generate optimization result chart.

        Args:
            baseline_results: Baseline result containing SQL path and building type.
            optimization_results: Optimization result containing SQL path and building type.
        """
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.15,
            ncols=1,
            nrows=5,
        )
        axs.format(
            abc="a",
            abcloc="ul",
            suptitle="Optimization EUI Comparison",
        )
        baseline_data = self.baseline_results.copy()
        optimization_data = self.optimization_results.copy()
        for i, building_type in enumerate(BUILDING_ORDER):
            weather_codes = optimization_data[building_type]
            improvement = {}
            for weather_code, optimization_result in weather_codes.items():
                baseline_result = baseline_data[building_type][weather_code]
                assert baseline_result.total_source_eui is not None
                assert optimization_result.total_source_eui is not None
                improvement[weather_code] = (
                    (
                        baseline_result.total_source_eui
                        - optimization_result.total_source_eui
                    )
                    / baseline_result.total_source_eui
                    * 100
                )
            sorted_improvement = {
                k: improvement[k] for k in WEATHER_ORDER if k in improvement
            }
            labels = list(sorted_improvement.keys())
            values = list(sorted_improvement.values())
            axs[i].bar(
                labels,
                values,
                color=self.style.get_color(i),
                bar_labels=True,
                bar_labels_kw={"fmt": "%.1f"},
            )
            axs[i].set_ylim(-10, 100)
            axs[i].set_yticks(np.arange(0, 100 + 1, 25))
            axs[i].set_title(BUILDING_NAME[building_type])
            axs[i].set_xlabel("Weather Code")
            axs[i].set_ylabel("Improvement (%)")

        self.save(
            fig,
            f"{Prefix.optimization}-Optimal Improvement Comparison",
            building_type=None,
        )

    def optimal_improvement_boxplot(self) -> None:
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.15,
            ncols=1,
        )
        axs.format(
            suptitle="Optimal Improvement Boxplot",
        )
        baseline_data = self.baseline_results.copy()
        optimization_data = self.optimization_results.copy()
        improvements: dict[str, list[float]] = defaultdict(list)
        for building_type in BUILDING_ORDER:
            weather_codes = optimization_data[building_type]
            for weather_code, optimization_result in weather_codes.items():
                baseline_result = baseline_data[building_type][weather_code]
                assert baseline_result.total_source_eui is not None
                assert optimization_result.total_source_eui is not None
                improvements[BUILDING_NAME[building_type]].append(
                    (
                        baseline_result.total_source_eui
                        - optimization_result.total_source_eui
                    )
                    / baseline_result.total_source_eui
                    * 100
                )

        df = pd.DataFrame(improvements)
        axs.box(df, marker="x", meancolor="r", fillcolor="gray4")
        axs.format(
            ylim=(0, 50),
            yticks=np.arange(0, 50 + 1, 10),
            ylabel="Improvement (%)",
            # xlabel="Building Type",
        )
        self.save(
            fig,
            f"{Prefix.optimization}-Optimal Improvement Boxplot",
            building_type=None,
        )

    def neutrality_timeline(self) -> None:
        df_a = pd.read_csv(self.csv_dir / "06_carbon_mode_a.csv")

        df_mid = df_a[df_a["cambium_scenario"] == "MidCase"].copy()
        df_avg = (
            df_mid.groupby(["building_type", "cambium_year"])
            .agg(carbon=("carbon_intensity_kgm2", "mean"))
            .reset_index()
        )

        df_all = (
            df_a.groupby(["building_type", "cambium_year"])
            .agg(
                carbon_min=("carbon_intensity_kgm2", "min"),
                carbon_max=("carbon_intensity_kgm2", "max"),
            )
            .reset_index()
        )

        df_bc = pd.read_csv(self.csv_dir / "05_carbon_mode_bc.csv")
        df_c_r1 = (
            df_bc[
                (df_bc["mode"] == "mode_c")
                & (df_bc["stage"] == "pv")
                & (df_bc["R"] == 1.0)
            ]
            .groupby("building_type")["carbon_gas_intensity_kgm2"]
            .mean()
        )

        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.55,
        )

        ymax = df_all["carbon_max"].max() * 1.15

        for building_type in BUILDING_ORDER:
            color = BUILDING_COLOR[building_type]
            sub = df_avg[df_avg["building_type"] == building_type].sort_values(
                "cambium_year"
            )
            sub_all = df_all[df_all["building_type"] == building_type].sort_values(
                "cambium_year"
            )

            ax.fill_between(
                sub_all["cambium_year"],
                sub_all["carbon_min"],
                sub_all["carbon_max"],
                color=color,
                alpha=0.12,
                label="_nolegend_",
            )
            ax.plot(
                sub["cambium_year"],
                sub["carbon"],
                color=color,
                linewidth=self.style.line_width,
                marker="o",
                markersize=5,
                label=BUILDING_NAME[building_type],
                zorder=5,
            )

            last = sub.iloc[-1]
            ax.text(
                last["cambium_year"] + 0.5,
                last["carbon"],
                f"{last['carbon']:.1f}",
                ha="left",
                va="center",
                fontsize=self.style.font_size_small,
                color=color,
            )

        for building_type in GAS_BUILDING_TYPE:
            color = BUILDING_COLOR[building_type]
            gas_value = df_c_r1[building_type]
            ax.axhline(
                gas_value,
                color=color,
                linestyle="--",
                linewidth=self.style.line_width,
                alpha=0.4,
            )
            ax.text(
                2051.2,
                gas_value + 0.2,
                f"Gas: {gas_value:.1f}",
                fontsize=self.style.font_size_small,
                color=color,
                alpha=0.8,
                ha="left",
                va="bottom",
            )

        ax.axhline(0, color="black", linewidth=self.style.line_width)
        ax.text(
            2024.5,
            0.5,
            "Carbon Neutral",
            fontsize=self.style.font_size,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="green",
        )

        policy = {
            2030: ("Illinois RPS\n40%", "grey"),
            2040: ("Illinois RPS\n50%", "grey"),
            2045: ("CEJA\n100% Clean", self.style.get_color(4)),
        }
        for year, (label, color) in policy.items():
            ax.axvline(
                year,
                color=color,
                linestyle=":",
                linewidth=self.style.line_width,
                alpha=0.6,
            )
            ax.text(
                year,
                ymax * 0.97,
                label,
                ha="center",
                va="top",
                fontsize=self.style.font_size_small,
                color=color,
                fontweight="bold" if year == 2045 else "normal",
            )

        ax.format(
            xlabel="Year",
            ylabel="Carbon Intensity (kg CO₂e/m²/yr)",
            xlim=(2024, 2053),
            ylim=(-10, ymax),
            title="Carbon Neutrality Pathway Timeline (Mode A, MidCase with Scenario Bands)",
            titleweight="bold",
            titlesize=self.style.font_size_title,
            labelsize=self.style.font_size_title,
        )
        ax.legend(loc="ur", ncol=1)

        self.save(fig, "Carbon Neutrality Pathway Timeline", building_type=None)

    def data_to_csv(self) -> None:
        data = []
        for building_type in BUILDING_ORDER:
            for weather_code in WEATHER_ORDER:
                baseline_result = self.baseline_results[building_type][weather_code]
                optimization_result = self.optimization_results[building_type][
                    weather_code
                ]
                pv_result = self.pv_results[building_type][weather_code]
                assert baseline_result.total_source_eui is not None
                assert optimization_result.total_source_eui is not None
                assert pv_result.total_source_eui is not None
                data.append(
                    self._extract_data(
                        building_type, weather_code, "baseline", baseline_result
                    )
                )
                data.append(
                    self._extract_data(
                        building_type, weather_code, "optimization", optimization_result
                    )
                )
                data.append(
                    self._extract_data(building_type, weather_code, "pv", pv_result)
                )
        df = pd.DataFrame(data)
        df.to_csv(self.paths.visualization_dir / "Results_data.csv", index=False)

    def _extract_data(
        self,
        building_type: str,
        weather_code: str,
        data_type: str,
        data: SimulationResult,
    ) -> dict:
        result = {}
        result["data_type"] = data_type
        result["weather_code"] = weather_code
        result["building_type"] = building_type

        result["total_building_area"] = data.total_building_area
        result["net_building_area"] = data.net_building_area
        result["total_source_energy"] = data.total_source_energy
        result["total_site_energy"] = data.total_site_energy
        result["net_site_energy"] = data.net_site_energy
        result["net_source_energy"] = data.net_source_energy
        result["total_source_eui"] = data.total_source_eui
        result["net_source_eui"] = data.net_source_eui
        result["total_site_eui"] = data.total_site_eui
        result["net_source_eui"] = data.net_source_eui
        result["net_site_eui"] = data.net_site_eui
        result["predicted_eui"] = data.predicted_eui
        return result

    def generate_all(self) -> None:
        # self.data_to_csv()
        # self.baseline_result()
        # self.optimal_improvement_comparison()
        # self.optimal_improvement_boxplot()
        # self.storage_soc()
        # self.typical_day_storage_soc()
        self.neutrality_timeline()
