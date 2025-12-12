"""
PV-related figures (F4-F6).

F4: Monthly PV generation statistics
F5: Climate scenario PV generation comparison
F6: Load-PV matching analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseFigure
from ..config import (
    FigureConfig,
    ColorSchemes,
    ALL_CLIMATES,
    ALL_BUILDINGS,
)


class MonthlyPVGenerationFigure(BaseFigure):
    """
    F4: Monthly PV generation statistics.

    Shows:
    - Monthly PV generation (bar chart)
    - Cumulative generation (line)
    - Monthly demand for comparison
    """

    def plot(
        self,
        df: pd.DataFrame,
        building: str,
        weather: str,
    ) -> plt.Figure:
        """
        Plot monthly PV generation.

        Args:
            df: Processed hourly DataFrame
            building: Building type
            weather: Weather scenario

        Returns:
            matplotlib Figure
        """
        fig, ax1 = self.create_figure(width="single", height=2.5)

        # Aggregate to monthly
        if "month" not in df.columns:
            return fig

        monthly = df.groupby("month").agg({
            "pv_generation_kw": "sum",
            "demand_kw": "sum",
        }).reindex(range(1, 13)).fillna(0)

        months = np.arange(1, 13)
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

        # Bar width
        width = 0.35

        # PV generation bars
        pv_bars = ax1.bar(
            months - width/2,
            monthly["pv_generation_kw"],
            width,
            color=ColorSchemes.ENERGY_COLORS["pv_generation"],
            alpha=0.8,
            label="PV Generation",
        )

        # Demand bars
        demand_bars = ax1.bar(
            months + width/2,
            monthly["demand_kw"],
            width,
            color=ColorSchemes.ENERGY_COLORS["demand"],
            alpha=0.8,
            label="Building Demand",
        )

        ax1.set_xlabel("Month")
        ax1.set_ylabel("Energy (kWh)")
        ax1.set_xticks(months)
        ax1.set_xticklabels(month_labels)

        # Secondary axis for cumulative generation
        ax2 = ax1.twinx()
        cumulative = monthly["pv_generation_kw"].cumsum()
        ax2.plot(
            months,
            cumulative,
            color=ColorSchemes.ENERGY_COLORS["self_consumed"],
            marker="s",
            markersize=3,
            linewidth=1.5,
            label="Cumulative PV",
        )
        ax2.set_ylabel("Cumulative Generation (kWh)")

        # Combined legend - moved outside to avoid overlap
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.15),
            ncol=3,
            frameon=False,
            fontsize=self.config.font_size_subscript,
        )

        # Title
        building_name = ColorSchemes.get_building_name(building)
        ax1.set_title(
            f"{building_name} - {weather}",
            fontsize=self.config.font_size_title,
        )

        # Annotations
        total_pv = monthly["pv_generation_kw"].sum()
        total_demand = monthly["demand_kw"].sum()
        coverage = (total_pv / total_demand * 100) if total_demand > 0 else 0

        ax1.annotate(
            f"Annual PV: {total_pv/1000:.1f} MWh\n"
            f"Annual Demand: {total_demand/1000:.1f} MWh\n"
            f"Coverage: {coverage:.1f}%",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            fontsize=self.config.font_size_subscript,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        fig.tight_layout()
        return fig


class ClimateComparisonFigure(BaseFigure):
    """
    F5: Climate scenario PV generation comparison.

    Shows annual PV generation across different climate scenarios.
    """

    def plot(
        self,
        annual_pv_dict: dict[str, float],
        building: str,
    ) -> plt.Figure:
        """
        Plot climate scenario comparison.

        Args:
            annual_pv_dict: Dict mapping weather scenario to annual PV (kWh)
            building: Building type

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=2.5)

        # Sort by climate scenario order
        weathers = [w for w in ALL_CLIMATES if w in annual_pv_dict]
        values = [annual_pv_dict[w] for w in weathers]
        colors = [ColorSchemes.get_climate_color(w) for w in weathers]

        # Bars
        bars = ax.bar(
            range(len(weathers)),
            np.array(values) / 1000,  # Convert to MWh
            color=colors,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val/1000:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=self.config.font_size_subscript,
            )

        # Baseline reference line (TMY)
        if "TMY" in annual_pv_dict:
            baseline = annual_pv_dict["TMY"] / 1000
            ax.axhline(
                y=baseline,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
                label=f"TMY Baseline ({baseline:.1f} MWh)",
            )

        self.format_axis(
            ax,
            xlabel="Climate Scenario",
            ylabel="Annual PV Generation (MWh)",
            legend=False,  # Disable auto legend
        )

        # Move legend outside to avoid overlap
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.2),
            frameon=False,
            fontsize=self.config.font_size_subscript,
        )

        ax.set_xticks(range(len(weathers)))
        ax.set_xticklabels(weathers, rotation=45, ha="right")

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - PV Generation by Climate",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig

    def plot_all_buildings(
        self,
        data: dict[str, dict[str, float]],
    ) -> plt.Figure:
        """
        Plot climate comparison for all buildings.

        Args:
            data: Nested dict {building: {weather: annual_pv}}

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=3.0)

        buildings = list(data.keys())
        weathers = ALL_CLIMATES
        n_buildings = len(buildings)
        n_weathers = len(weathers)

        # Bar positions
        x = np.arange(n_buildings)
        width = 0.12

        # Plot bars for each weather scenario
        for i, weather in enumerate(weathers):
            values = [data[b].get(weather, 0) / 1000 for b in buildings]
            offset = (i - n_weathers/2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                color=ColorSchemes.get_climate_color(weather),
                alpha=0.8,
                label=weather,
            )

        self.format_axis(
            ax,
            xlabel="Building Type",
            ylabel="Annual PV Generation (MWh)",
            legend=True,
            legend_loc="upper right",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [ColorSchemes.get_building_name(b) for b in buildings],
            rotation=45,
            ha="right",
        )

        ax.set_title(
            "PV Generation Comparison by Building and Climate",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class LoadPVMatchFigure(BaseFigure):
    """
    F6: Building load and PV generation matching analysis.

    Shows:
    - Hourly load profile
    - Hourly PV generation profile
    - Excess/deficit visualization
    """

    def plot(
        self,
        df: pd.DataFrame,
        building: str,
        weather: str,
        day_range: tuple[int, int] = (1, 7),
    ) -> plt.Figure:
        """
        Plot load-PV matching for a week.

        Args:
            df: Processed hourly DataFrame
            building: Building type
            weather: Weather scenario
            day_range: Day of year range to plot (start, end)

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=2.5)

        # Filter to specified days
        start_day, end_day = day_range
        mask = (df["dayofyear"] >= start_day) & (df["dayofyear"] <= end_day)
        week_df = df[mask].copy()

        if len(week_df) == 0:
            ax.text(0.5, 0.5, "No data for specified period", ha="center", va="center")
            return fig

        hours = np.arange(len(week_df))

        # PV generation area
        if "pv_generation_kw" in week_df.columns:
            ax.fill_between(
                hours,
                0,
                week_df["pv_generation_kw"],
                alpha=0.6,
                color=ColorSchemes.ENERGY_COLORS["pv_generation"],
                label="PV Generation",
            )

        # Demand line
        if "demand_kw" in week_df.columns:
            ax.plot(
                hours,
                week_df["demand_kw"],
                color=ColorSchemes.ENERGY_COLORS["demand"],
                linewidth=1.0,
                label="Building Demand",
            )

        # Highlight excess (PV > demand) - use orange color for distinction
        if "excess_pv_kw" in week_df.columns:
            excess = week_df["excess_pv_kw"]
            ax.fill_between(
                hours,
                week_df["demand_kw"],
                week_df["demand_kw"] + excess,
                where=excess > 0,
                alpha=0.7,
                color="#ff7f0e",  # Orange for better distinction from PV yellow
                label="Excess PV",
                hatch="//",
            )

        # Highlight deficit (demand > PV)
        if "deficit_kw" in week_df.columns:
            deficit = week_df["deficit_kw"]
            pv = week_df.get("pv_generation_kw", 0)
            ax.fill_between(
                hours,
                pv,
                week_df["demand_kw"],
                where=deficit > 0,
                alpha=0.4,
                color=ColorSchemes.ENERGY_COLORS["grid_import"],
                label="Grid Import",
            )

        self.format_axis(
            ax,
            xlabel="Hour",
            ylabel="Power (kW)",
            legend=False,  # Disable auto legend
        )

        # Move legend outside to avoid overlap
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.12),
            ncol=4,
            frameon=False,
            fontsize=self.config.font_size_subscript,
        )

        # Day markers
        for d in range(end_day - start_day + 1):
            ax.axvline(x=d * 24, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - {weather} (Days {start_day}-{end_day})",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig

    def plot_average_day(
        self,
        df: pd.DataFrame,
        building: str,
        weather: str,
    ) -> plt.Figure:
        """
        Plot average daily load-PV profile.

        Args:
            df: Processed hourly DataFrame
            building: Building type
            weather: Weather scenario

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=2.5)

        # Calculate hourly averages
        hourly_avg = df.groupby("hour").agg({
            "pv_generation_kw": "mean",
            "demand_kw": "mean",
        })

        hours = np.arange(24)

        # PV area
        ax.fill_between(
            hours,
            0,
            hourly_avg["pv_generation_kw"],
            alpha=0.6,
            color=ColorSchemes.ENERGY_COLORS["pv_generation"],
            label="Avg PV Generation",
        )

        # Demand line
        ax.plot(
            hours,
            hourly_avg["demand_kw"],
            color=ColorSchemes.ENERGY_COLORS["demand"],
            linewidth=1.5,
            label="Avg Demand",
        )

        self.format_axis(
            ax,
            xlabel="Hour of Day",
            ylabel="Power (kW)",
            xlim=(-0.5, 23.5),
            legend=True,
            legend_loc="upper right",
        )

        ax.set_xticks([0, 6, 12, 18, 24])

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - {weather}\nAverage Daily Profile",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class PVPerformanceSummaryFigure(BaseFigure):
    """
    Summary figure for PV performance across all combinations.
    """

    def plot(
        self,
        summary_df: pd.DataFrame,
    ) -> plt.Figure:
        """
        Plot PV performance summary heatmap.

        Args:
            summary_df: DataFrame with building, weather, and metrics

        Returns:
            matplotlib Figure
        """
        fig, axes = self.create_figure(
            width="double",
            height=4.0,
            nrows=2,
            ncols=2,
        )

        metrics = [
            ("total_pv_kwh", "Annual PV Generation (kWh)"),
            ("scr", "Self-Consumption Rate (%)"),
            ("ssr", "Self-Sufficiency Rate (%)"),
            ("curtailment_rate", "Curtailment Rate (%)"),
        ]

        for ax, (metric, title) in zip(axes.flat, metrics):
            if metric not in summary_df.columns:
                ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
                continue

            # Pivot for heatmap
            pivot = summary_df.pivot(
                index="building",
                columns="weather",
                values=metric,
            )

            # Reorder
            pivot = pivot.reindex(
                index=[b for b in ALL_BUILDINGS if b in pivot.index],
                columns=[w for w in ALL_CLIMATES if w in pivot.columns],
            )

            # Heatmap
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="YlOrRd" if "curtailment" not in metric else "YlOrRd_r",
                annot=True,
                fmt=".1f",
                annot_kws={"size": self.config.font_size_subscript},
                cbar_kws={"shrink": 0.8},
            )

            ax.set_title(title, fontsize=self.config.font_size_title)
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Rotate labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(
                [ColorSchemes.get_building_name(t.get_text()) for t in ax.get_yticklabels()],
                rotation=0,
            )

        fig.tight_layout()
        return fig
