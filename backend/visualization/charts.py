from datetime import datetime
from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd
import ultraplot as uplt
from sqlalchemy import create_engine

from backend.models import BuildingType, SimulationResult
from backend.utils.config import ConfigManager


def hour_of_year(dt: datetime) -> int:
    year_start = datetime(2017, 1, 1)
    delta = dt - year_start
    return int(delta.total_seconds() // 3600)


class ChartGenerator:
    STORAGE_SOC_QUERY = """
            SELECT
                keyvalue,
                month,
                day,
                hour,
                value,
                interval,
                units
            FROM
                `ReportVariableWithTime`
            WHERE name = 'Electric Storage Simple Charge State'
            """

    TYPICAL_SUMMER_DAY_SOC_QUERY = """
                SELECT
            month,
                day,
                hour,
                sum(case when name = 'Generator Produced DC Electricity Rate' then value else 0 end) as pv_value,
                sum(case when name = 'Electric Storage Charge Power' then value else 0 end) as storage_charge_value,
                sum(case when name = 'Electric Storage Discharge Power' then value else 0 end) as storage_discharge_value,
                sum(case when name = 'Facility Total Electricity Demand Rate' then value else 0 end) as demand_value
            FROM
                `ReportVariableWithTime`
            WHERE
                name in ('Generator Produced DC Electricity Rate', 'Electric Storage Charge Power', 'Electric Storage Discharge Power', 'Facility Total Electricity Demand Rate') and month = 7 and day in (12,13,14,15,16,17,18)
            GROUP BY
                month, day, hour
    """

    TYPICAL_WINTER_DAY_SOC_QUERY = """
                SELECT
            month,
                day,
                hour,
                sum(case when name = 'Generator Produced DC Electricity Rate' then value else 0 end) as pv_value,
                sum(case when name = 'Electric Storage Charge Power' then value else 0 end) as storage_charge_value,
                sum(case when name = 'Electric Storage Discharge Power' then value else 0 end) as storage_discharge_value,
                sum(case when name = 'Facility Total Electricity Demand Rate' then value else 0 end) as demand_value
            FROM
                `ReportVariableWithTime`
            WHERE
                name in ('Generator Produced DC Electricity Rate', 'Electric Storage Charge Power', 'Electric Storage Discharge Power', 'Facility Total Electricity Demand Rate') and month = 1 and day in (12,13,14,15,16,17,18)
            GROUP BY
                month, day, hour
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        self.paths = config.paths
        self.output_dir = config.paths.visualization_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        uplt.rc.update(
            {
                "font.family": "serif",
                "font.size": 8,
                "savefig.dpi": 300,
            }
        )
        self.pv_results = self._pv_data_prepare()

    def _pv_data_prepare(self) -> list[SimulationResult]:
        pv_results = []
        pv_dir = self.paths.pv_dir
        for pv_file in pv_dir.glob("**/result.pkl"):
            with open(pv_file, "rb") as f:
                pv_data = load(f)
                pv_results.append(pv_data)
        return pv_results

    def save(self, fig: uplt.Figure, building_type: BuildingType, name: str) -> Path:
        image_dir = self.output_dir / building_type.value
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{name}.png"
        fig.save(path)
        return path

    def storage_soc(self, pv_result: SimulationResult) -> None:
        fig, ax = uplt.subplots(refwidth=6, refheight=2)

        try:
            weather_code = (
                pv_result.sql_path.parent.name
                if pv_result.sql_path is not None
                else "Unknown"
            )
            building_type = pv_result.building_type
            capacity = self.config.storage.capacity[pv_result.building_type]

            engine = create_engine(f"sqlite:///{pv_result.sql_path}")  # type: ignore
            soc_data = pd.read_sql_query(self.STORAGE_SOC_QUERY, engine)
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
                    linewidth=0.3,
                    alpha=0.8,
                    color="#9b59b6",
                )
                ax.set_xlabel("Hour of Year")
                ax.set_ylabel("State of Charge (%)")
                ax.set_title(f"{weather_code} - {building_type}")

                ax.axhline(
                    y=100,
                    color="#27ae60",
                    linestyle="--",
                    linewidth=1.0,
                    label="Full capacity (100%)",
                )
                ax.axhline(
                    y=50,
                    color="#aaaaaa",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.5,
                    label="Half capacity (50%)",
                )

                ax.set_ylim(0, 110)
                ax.set_yticks(np.arange(0, 110, 20))
                ax.set_xlim(0, 8760)
                ax.set_xticks(np.arange(0, 8760, 1000))

                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0, -0.2),
                    ncol=3,
                    frameon=False,
                )

                ax.annotate(
                    f"Capacity: {capacity} kWh",
                    xy=(0.02, 0.98),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                    fontsize=8,
                )

                self.save(
                    fig,
                    building_type,
                    f"F01-{weather_code}-{building_type}-{keyvalue}-Storage SOC",
                )
        finally:
            uplt.close(fig)

    def typical_day_storage_soc(self, pv_result: SimulationResult) -> None:
        fig, axs = uplt.subplots(ncols=2, refwidth=4, refheight=2, sharey=True)
        titles = ["Summer (July 12-18)", "Winter Day (January 12-18)"]
        querys = [self.TYPICAL_SUMMER_DAY_SOC_QUERY, self.TYPICAL_WINTER_DAY_SOC_QUERY]

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
                linewidth=1.0,
                color="#34495e",
                label="Demand",
            )
            ax.fill_between(
                hours, pv, alpha=0.6, color="#f1c40f", label="PV Generation"
            )
            ax.bar(
                hours,
                storage_charge,
                width=0.6,
                alpha=0.7,
                color="#9b59b6",
                label="Storage Charge",
            )
            ax.bar(
                hours,
                storage_discharge,
                width=0.6,
                alpha=0.7,
                color="#1abc9c",
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
            axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False)

            engine = create_engine(f"sqlite:///{pv_result.sql_path}")  # type: ignore
            for ax, query, title in zip(axs, querys, titles, strict=True):
                typical_df = pd.read_sql_query(query, engine)
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
            # self.storage_soc(pv_result)
            self.typical_day_storage_soc(pv_result)
