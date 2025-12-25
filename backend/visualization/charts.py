"""
Minimal chart generation module using ultraplot.

Provides publication-quality figures for energy simulation results.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ultraplot as uplt

# Color schemes
CLIMATE_COLORS = {
    "TMY": "#1f77b4",
    "SSP126": "#2ca02c",
    "SSP245": "#9467bd",
    "SSP370": "#ff7f0e",
    "SSP434": "#d62728",
    "SSP585": "#8c564b",
}

ENERGY_COLORS = {
    "pv": "#f1c40f",
    "demand": "#34495e",
    "storage_charge": "#9b59b6",
    "storage_discharge": "#1abc9c",
    "curtailed": "#95a5a6",
}

BUILDING_NAMES = {
    "OfficeLarge": "Large Office",
    "OfficeMedium": "Medium Office",
    "ApartmentHighRise": "High-Rise Apartment",
    "SingleFamilyResidential": "Single Family",
    "MultiFamilyResidential": "Multi-Family",
}


class ChartGenerator:
    """Generate publication-quality charts using ultraplot."""

    def __init__(self, output_dir: Path | str = "output/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        uplt.rc.update({"font.size": 8, "savefig.dpi": 300})

    def save(self, fig: uplt.Figure, name: str) -> Path:
        """Save figure to file."""
        path = self.output_dir / f"{name}.png"
        fig.save(path)
        return path

    def storage_soc(
        self, df: pd.DataFrame, building: str, weather: str
    ) -> uplt.Figure:
        """Plot storage state of charge over the year."""
        fig, ax = uplt.subplots(refwidth=6, refheight=2)
        soc = df.get("storage_soc_pct", pd.Series(dtype=float))
        if soc.empty:
            ax.format(title=f"No storage data - {building}")
            return fig

        ax.plot(soc.values, lw=0.3, color=ENERGY_COLORS["storage_charge"])
        ax.axhline(100, ls="--", lw=1, color="red", label="Full")
        ax.format(
            title=f"{BUILDING_NAMES.get(building, building)} - {weather}",
            xlabel="Hour of year",
            ylabel="SOC (%)",
            xlim=(0, len(soc)),
            ylim=(0, 105),
        )
        ax.legend(loc="ur", ncols=1)
        return fig

    def typical_day(
        self,
        summer_df: pd.DataFrame,
        winter_df: pd.DataFrame,
        building: str,
    ) -> uplt.Figure:
        """Plot typical summer and winter day comparison."""
        fig, axs = uplt.subplots(ncols=2, refwidth=3, refheight=2, share=False)

        for ax, day_df, season in zip(
            axs, [summer_df, winter_df], ["Summer", "Winter"]
        ):
            hours = np.arange(min(24, len(day_df)))
            if "pv_generation_kw" in day_df.columns:
                pv = day_df["pv_generation_kw"].values[:24]
                ax.area(hours, pv, alpha=0.6, color=ENERGY_COLORS["pv"], label="PV")
            if "demand_kw" in day_df.columns:
                demand = day_df["demand_kw"].values[:24]
                ax.plot(hours, demand, lw=1.5, color=ENERGY_COLORS["demand"], label="Demand")
            ax.format(title=season, xlabel="Hour", ylabel="Power (kW)", xlim=(0, 23))

        axs[0].legend(loc="ur", ncols=1)
        fig.format(suptitle=BUILDING_NAMES.get(building, building))
        return fig

    def monthly_pv(
        self, df: pd.DataFrame, building: str, weather: str
    ) -> uplt.Figure:
        """Plot monthly PV generation vs demand."""
        fig, ax = uplt.subplots(refwidth=4, refheight=2.5)

        if "month" not in df.columns:
            ax.format(title="No monthly data")
            return fig

        monthly = df.groupby("month").agg(
            {"pv_generation_kw": "sum", "demand_kw": "sum"}
        ).reindex(range(1, 13)).fillna(0)

        months = np.arange(1, 13)
        width = 0.35
        ax.bar(
            months - width / 2,
            monthly["pv_generation_kw"] / 1000,
            width=width,
            color=ENERGY_COLORS["pv"],
            label="PV (MWh)",
        )
        ax.bar(
            months + width / 2,
            monthly["demand_kw"] / 1000,
            width=width,
            color=ENERGY_COLORS["demand"],
            label="Demand (MWh)",
        )
        ax.format(
            title=f"{BUILDING_NAMES.get(building, building)} - {weather}",
            xlabel="Month",
            ylabel="Energy (MWh)",
            xlocator=months,
            xformatter=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
        )
        ax.legend(loc="ur", ncols=1)
        return fig

    def climate_comparison(
        self, annual_pv: dict[str, float], building: str
    ) -> uplt.Figure:
        """Compare annual PV generation across climate scenarios."""
        fig, ax = uplt.subplots(refwidth=4, refheight=2.5)

        weathers = list(annual_pv.keys())
        values = [annual_pv[w] / 1000 for w in weathers]
        colors = [CLIMATE_COLORS.get(w, "#333") for w in weathers]

        ax.bar(range(len(weathers)), values, color=colors, edgecolor="white")

        # Add TMY reference line
        if "TMY" in annual_pv:
            ax.axhline(annual_pv["TMY"] / 1000, ls="--", lw=1, color="gray")

        ax.format(
            title=BUILDING_NAMES.get(building, building),
            xlabel="Climate Scenario",
            ylabel="Annual PV (MWh)",
            xlocator=range(len(weathers)),
            xformatter=weathers,
        )
        return fig

    def eui_waterfall(
        self,
        baseline: float,
        optimized: float,
        pv: float,
        building: str,
    ) -> uplt.Figure:
        """Plot EUI reduction waterfall chart."""
        fig, ax = uplt.subplots(refwidth=4, refheight=2.5)

        stages = ["Baseline", "ECM", "PV+Storage"]
        values = [baseline, optimized, pv]
        colors = ["#e74c3c", "#f39c12", "#27ae60"]

        ax.bar(range(len(stages)), values, color=colors, edgecolor="white", width=0.6)

        # Add reduction annotations
        for i, (v, s) in enumerate(zip(values, stages)):
            ax.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=8)

        ax.format(
            title=f"{BUILDING_NAMES.get(building, building)} - EUI Reduction",
            xlabel="Stage",
            ylabel="Site EUI (kWh/m\u00b2)",
            xlocator=range(len(stages)),
            xformatter=stages,
            ylim=(0, max(values) * 1.15),
        )
        return fig

    def load_pv_match(
        self, df: pd.DataFrame, building: str, day_start: int = 182
    ) -> uplt.Figure:
        """Plot load and PV matching for a sample week."""
        fig, ax = uplt.subplots(refwidth=6, refheight=2.5)

        # Extract one week of data
        start_hour = day_start * 24
        end_hour = min(start_hour + 168, len(df))
        week_df = df.iloc[start_hour:end_hour]
        hours = np.arange(len(week_df))

        if "pv_generation_kw" in week_df.columns:
            ax.area(
                hours,
                week_df["pv_generation_kw"].values,
                alpha=0.6,
                color=ENERGY_COLORS["pv"],
                label="PV Generation",
            )
        if "demand_kw" in week_df.columns:
            ax.plot(
                hours,
                week_df["demand_kw"].values,
                lw=1,
                color=ENERGY_COLORS["demand"],
                label="Demand",
            )

        ax.format(
            title=f"{BUILDING_NAMES.get(building, building)} - Summer Week",
            xlabel="Hour",
            ylabel="Power (kW)",
            xlim=(0, len(hours)),
        )
        ax.legend(loc="ur", ncols=1)
        return fig

    def performance_radar(self, metrics: dict[str, dict[str, float]]) -> uplt.Figure:
        """Plot radar chart comparing building performance."""
        fig, ax = uplt.subplots(proj="polar", refwidth=3)

        categories = list(next(iter(metrics.values())).keys())
        n = len(categories)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        for building, data in metrics.items():
            values = [data.get(c, 0) for c in categories]
            values += values[:1]
            ax.plot(angles, values, lw=1.5, label=BUILDING_NAMES.get(building, building))
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=7)
        ax.legend(loc="ur", ncols=1)
        fig.format(suptitle="Performance Comparison")
        return fig

    def sensitivity_heatmap(
        self, df: pd.DataFrame, target: str = "total_site_eui"
    ) -> uplt.Figure:
        """Plot ECM parameter sensitivity heatmap."""
        fig, ax = uplt.subplots(refwidth=5, refheight=4)

        # Calculate correlation with target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ecm_cols = [c for c in numeric_cols if c not in [target, "building_type"]]

        if not ecm_cols or target not in df.columns:
            ax.format(title="Insufficient data for sensitivity analysis")
            return fig

        correlations = df[ecm_cols].corrwith(df[target]).abs().sort_values(ascending=True)

        # Horizontal bar chart
        y_pos = np.arange(len(correlations))
        ax.barh(y_pos, correlations.values, color="steelblue", height=0.7)
        ax.format(
            title=f"ECM Sensitivity to {target}",
            xlabel="Absolute Correlation",
            ylabel="Parameter",
            ylocator=y_pos,
            yformatter=correlations.index.tolist(),
            xlim=(0, 1),
        )
        return fig


def generate_all_figures(
    data_dir: Path | str,
    output_dir: Path | str = "output/figures",
) -> list[Path]:
    """Generate all visualization figures from simulation data."""
    from loguru import logger

    data_dir = Path(data_dir)
    gen = ChartGenerator(output_dir)
    saved = []

    buildings = ["OfficeLarge", "OfficeMedium", "SingleFamilyResidential"]
    climates = ["TMY", "SSP126", "SSP585"]

    for building in buildings:
        for weather in climates:
            csv_path = data_dir / "pv" / building / weather / "pv_out.csv"
            if not csv_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
                df["month"] = pd.to_datetime(df.get("timestamp", df.index)).dt.month

                # Generate figures
                fig = gen.monthly_pv(df, building, weather)
                saved.append(gen.save(fig, f"monthly_pv_{building}_{weather}"))

                fig = gen.load_pv_match(df, building)
                saved.append(gen.save(fig, f"load_match_{building}_{weather}"))

                if "storage_soc_pct" in df.columns:
                    fig = gen.storage_soc(df, building, weather)
                    saved.append(gen.save(fig, f"storage_soc_{building}_{weather}"))

                logger.info(f"Generated figures for {building}/{weather}")
            except Exception as e:
                logger.warning(f"Failed {building}/{weather}: {e}")

    return saved
