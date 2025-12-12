"""
Energy analysis figures (F9-F12).

F9: EUI waterfall chart (Baseline → Optimization → PV)
F10: Climate scenario EUI trend
F11: Self-consumption rate analysis
F12: Peak load reduction analysis
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


class EUIWaterfallFigure(BaseFigure):
    """
    F9: EUI change waterfall chart.

    Shows progressive EUI reduction:
    Baseline → ECM Optimization → PV Integration → Final
    """

    def plot(
        self,
        baseline_eui: float,
        optimization_eui: float,
        pv_eui: float,
        building: str,
        weather: str,
    ) -> plt.Figure:
        """
        Plot EUI waterfall chart.

        Args:
            baseline_eui: Baseline EUI (kWh/m²/yr)
            optimization_eui: Post-optimization EUI
            pv_eui: Post-PV integration EUI
            building: Building type
            weather: Weather scenario

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=3.0)

        # Calculate changes
        ecm_reduction = baseline_eui - optimization_eui
        pv_reduction = optimization_eui - pv_eui

        # Data for waterfall
        stages = ["Baseline", "ECM\nOptimization", "PV\nIntegration", "Final"]
        values = [baseline_eui, -ecm_reduction, -pv_reduction, pv_eui]

        # Cumulative values for bar positions
        cumulative = [baseline_eui, optimization_eui, pv_eui, pv_eui]

        # Colors
        colors = [
            ColorSchemes.BUILDING_COLORS.get(building, "#3498db"),  # Baseline
            ColorSchemes.ENERGY_COLORS["self_consumed"],  # ECM reduction (green)
            ColorSchemes.ENERGY_COLORS["pv_generation"],  # PV reduction (yellow)
            ColorSchemes.ENERGY_COLORS["storage_charge"],  # Final (purple)
        ]

        # Plot bars
        bar_width = 0.6
        x_positions = np.arange(len(stages))

        for i, (stage, val, cum, color) in enumerate(
            zip(stages, values, cumulative, colors)
        ):
            if i == 0:
                # Baseline - full bar from 0
                ax.bar(i, cum, bar_width, color=color, alpha=0.8, edgecolor="white")
            elif i == len(stages) - 1:
                # Final - full bar from 0
                ax.bar(i, cum, bar_width, color=color, alpha=0.8, edgecolor="white")
            else:
                # Reduction bars - floating
                bottom = cumulative[i]
                height = -val
                ax.bar(
                    i, height, bar_width,
                    bottom=bottom,
                    color=color,
                    alpha=0.8,
                    edgecolor="white",
                )

        # Connecting lines
        for i in range(len(stages) - 1):
            ax.plot(
                [i + bar_width/2, i + 1 - bar_width/2],
                [cumulative[i], cumulative[i]],
                "k--",
                linewidth=0.8,
                alpha=0.5,
            )

        # Value labels
        ax.text(0, baseline_eui + 2, f"{baseline_eui:.1f}",
                ha="center", va="bottom", fontsize=self.config.font_size_subscript)
        ax.text(1, optimization_eui - ecm_reduction/2, f"-{ecm_reduction:.1f}",
                ha="center", va="center", fontsize=self.config.font_size_subscript,
                color="white", fontweight="bold")
        ax.text(2, pv_eui - pv_reduction/2, f"-{pv_reduction:.1f}",
                ha="center", va="center", fontsize=self.config.font_size_subscript,
                fontweight="bold")
        ax.text(3, pv_eui + 2, f"{pv_eui:.1f}",
                ha="center", va="bottom", fontsize=self.config.font_size_subscript)

        # Format
        ax.set_xticks(x_positions)
        ax.set_xticklabels(stages, fontsize=self.config.font_size_subscript)
        ax.set_ylabel("Site EUI (kWh/m²/yr)")
        ax.set_ylim(0, baseline_eui * 1.15)

        # Reduction summary
        total_reduction = baseline_eui - pv_eui
        total_pct = (total_reduction / baseline_eui * 100) if baseline_eui > 0 else 0

        ax.annotate(
            f"Total Reduction: {total_reduction:.1f} kWh/m²/yr ({total_pct:.1f}%)",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            fontsize=self.config.font_size_subscript,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - {weather}",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig

    def plot_all_buildings(
        self,
        eui_data: dict[str, dict[str, float]],
        weather: str = "TMY",
    ) -> plt.Figure:
        """
        Plot waterfall for all buildings.

        Args:
            eui_data: {building: {baseline, optimization, pv}}
            weather: Weather scenario

        Returns:
            matplotlib Figure
        """
        buildings = list(eui_data.keys())
        n_buildings = len(buildings)

        fig, axes = self.create_figure(
            width="double",
            height=2.0 * ((n_buildings + 1) // 2),
            nrows=(n_buildings + 1) // 2,
            ncols=2,
        )
        axes = axes.flat if n_buildings > 1 else [axes]

        for ax, building in zip(axes, buildings):
            data = eui_data[building]
            self._plot_mini_waterfall(
                ax,
                data.get("baseline", 0),
                data.get("optimization", 0),
                data.get("pv", 0),
                building,
            )

        # Hide unused axes
        for ax in axes[n_buildings:]:
            ax.set_visible(False)

        fig.suptitle(
            f"EUI Reduction Summary - {weather}",
            fontsize=self.config.font_size_title,
        )
        fig.tight_layout()
        return fig

    def _plot_mini_waterfall(
        self,
        ax: plt.Axes,
        baseline: float,
        optimization: float,
        pv: float,
        building: str,
    ) -> None:
        """Plot simplified waterfall on given axes."""
        stages = ["Base", "ECM", "PV", "Final"]
        cumulative = [baseline, optimization, pv, pv]
        colors = ["#3498db", "#27ae60", "#f1c40f", "#9b59b6"]

        x = np.arange(len(stages))
        ax.bar(x, cumulative, color=colors, alpha=0.8, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=self.config.font_size_subscript)
        ax.set_ylabel("EUI")

        reduction_pct = ((baseline - pv) / baseline * 100) if baseline > 0 else 0
        ax.set_title(
            f"{ColorSchemes.get_building_name(building)} (-{reduction_pct:.0f}%)",
            fontsize=self.config.font_size_subscript,
        )


class ClimateEUITrendFigure(BaseFigure):
    """
    F10: Climate scenario EUI trend.

    Shows how EUI changes across different climate scenarios.
    """

    def plot(
        self,
        eui_data: dict[str, dict[str, float]],
        building: str,
        eui_type: str = "total_site_eui",
    ) -> plt.Figure:
        """
        Plot EUI trend across climate scenarios.

        Args:
            eui_data: {weather: {baseline, optimization, pv}}
            building: Building type
            eui_type: Type of EUI to plot

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=2.5)

        weathers = [w for w in ALL_CLIMATES if w in eui_data]
        x = np.arange(len(weathers))
        width = 0.25

        # Extract values
        baseline_vals = [eui_data[w].get("baseline", 0) for w in weathers]
        opt_vals = [eui_data[w].get("optimization", 0) for w in weathers]
        pv_vals = [eui_data[w].get("pv", 0) for w in weathers]

        # Plot grouped bars
        ax.bar(x - width, baseline_vals, width, label="Baseline",
               color=ColorSchemes.ENERGY_COLORS["demand"], alpha=0.8)
        ax.bar(x, opt_vals, width, label="Optimized",
               color=ColorSchemes.ENERGY_COLORS["self_consumed"], alpha=0.8)
        ax.bar(x + width, pv_vals, width, label="With PV",
               color=ColorSchemes.ENERGY_COLORS["pv_generation"], alpha=0.8)

        # TMY reference line
        if "TMY" in eui_data:
            tmy_baseline = eui_data["TMY"].get("baseline", 0)
            ax.axhline(y=tmy_baseline, color="black", linestyle="--",
                      linewidth=0.8, alpha=0.5, label="TMY Baseline")

        self.format_axis(
            ax,
            xlabel="Climate Scenario",
            ylabel="Net Site EUI (kWh/m²/yr)",
            legend=True,
            legend_loc="upper left",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(weathers, rotation=45, ha="right")

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - Climate Impact on Net EUI",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig

    def plot_all_buildings(
        self,
        data: dict[str, dict[str, dict[str, float]]],
        stage: str = "pv",
    ) -> plt.Figure:
        """
        Plot EUI trend for all buildings.

        Args:
            data: {building: {weather: {baseline, optimization, pv}}}
            stage: Which stage to plot (baseline, optimization, pv)

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=3.0)

        buildings = list(data.keys())
        weathers = ALL_CLIMATES

        for building in buildings:
            if building not in data:
                continue

            values = []
            available_weathers = []
            for w in weathers:
                if w in data[building]:
                    values.append(data[building][w].get(stage, 0))
                    available_weathers.append(w)

            if values:
                ax.plot(
                    available_weathers,
                    values,
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    color=ColorSchemes.get_building_color(building),
                    label=ColorSchemes.get_building_name(building),
                )

        self.format_axis(
            ax,
            xlabel="Climate Scenario",
            ylabel="Site EUI (kWh/m²/yr)",
            legend=True,
            legend_loc="upper left",
        )

        stage_labels = {
            "baseline": "Baseline",
            "optimization": "Optimized",
            "pv": "With PV",
        }
        ax.set_title(
            f"EUI Trend by Climate - {stage_labels.get(stage, stage)}",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class SelfConsumptionFigure(BaseFigure):
    """
    F11: Self-consumption rate (SCR) vs Self-sufficiency rate (SSR) analysis.
    """

    def plot(
        self,
        metrics_data: dict[str, dict[str, dict]],
    ) -> plt.Figure:
        """
        Plot SCR vs SSR scatter.

        Args:
            metrics_data: {building: {weather: {scr, ssr, ...}}}

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=3.0)

        for building, weathers in metrics_data.items():
            scr_vals = []
            ssr_vals = []

            for weather, metrics in weathers.items():
                if "scr" in metrics and "ssr" in metrics:
                    scr_vals.append(metrics["scr"])
                    ssr_vals.append(metrics["ssr"])

            if scr_vals:
                ax.scatter(
                    scr_vals,
                    ssr_vals,
                    c=ColorSchemes.get_building_color(building),
                    label=ColorSchemes.get_building_name(building),
                    alpha=0.7,
                    s=30,
                    edgecolors="white",
                    linewidth=0.5,
                )

        # Reference lines
        ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        # Diagonal reference (SCR = SSR)
        ax.plot([0, 100], [0, 100], "k:", linewidth=0.5, alpha=0.3)

        self.format_axis(
            ax,
            xlabel="Self-Consumption Rate (%)",
            ylabel="Self-Sufficiency Rate (%)",
            xlim=(0, 105),
            ylim=(0, 105),
            legend=True,
            legend_loc="upper right",
        )

        # Quadrant labels
        ax.text(75, 25, "High SCR\nLow SSR", ha="center", va="center",
                fontsize=self.config.font_size_subscript, alpha=0.5)
        ax.text(25, 75, "Low SCR\nHigh SSR", ha="center", va="center",
                fontsize=self.config.font_size_subscript, alpha=0.5)

        ax.set_title(
            "Self-Consumption vs Self-Sufficiency",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class PeakReductionFigure(BaseFigure):
    """
    F12: Peak load reduction analysis.

    Shows peak demand reduction from PV integration.
    """

    def plot(
        self,
        peak_data: dict[str, dict[str, dict]],
    ) -> plt.Figure:
        """
        Plot peak reduction boxplot.

        Args:
            peak_data: {building: {weather: {baseline_peak, pv_peak, reduction_pct}}}

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=3.0)

        # Prepare data for boxplot
        buildings = []
        reductions = []

        for building, weathers in peak_data.items():
            building_reductions = []
            for weather, metrics in weathers.items():
                if "peak_reduction_pct" in metrics:
                    building_reductions.append(metrics["peak_reduction_pct"])

            if building_reductions:
                buildings.append(building)
                reductions.append(building_reductions)

        if not buildings:
            ax.text(0.5, 0.5, "No peak reduction data", ha="center", va="center")
            return fig

        # Boxplot
        bp = ax.boxplot(
            reductions,
            labels=[ColorSchemes.get_building_name(b) for b in buildings],
            patch_artist=True,
        )

        # Color boxes
        for patch, building in zip(bp["boxes"], buildings):
            patch.set_facecolor(ColorSchemes.get_building_color(building))
            patch.set_alpha(0.7)

        # Add mean markers
        means = [np.mean(r) for r in reductions]
        ax.scatter(
            range(1, len(buildings) + 1),
            means,
            marker="D",
            color="black",
            s=30,
            zorder=3,
            label="Mean",
        )

        # Zero reference line
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        self.format_axis(
            ax,
            ylabel="Peak Load Reduction (%)",
            legend=True,
            legend_loc="upper right",
        )

        ax.set_xticklabels(
            [ColorSchemes.get_building_name(b) for b in buildings],
            rotation=45,
            ha="right",
        )

        ax.set_title(
            "Peak Load Reduction by Building Type",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig

    def plot_monthly(
        self,
        df: pd.DataFrame,
        building: str,
        weather: str,
    ) -> plt.Figure:
        """
        Plot monthly peak reduction.

        Args:
            df: Processed hourly DataFrame
            building: Building type
            weather: Weather scenario

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=2.5)

        if "month" not in df.columns or "demand_kw" not in df.columns:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        # Calculate monthly peaks
        monthly_demand_peak = df.groupby("month")["demand_kw"].max()

        monthly_net_peak = None
        if "net_demand_kw" in df.columns:
            monthly_net_peak = df.groupby("month")["net_demand_kw"].max()

        months = np.arange(1, 13)
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

        width = 0.35

        # Demand peak bars
        ax.bar(
            months - width/2,
            monthly_demand_peak.reindex(months).fillna(0),
            width,
            label="Demand Peak",
            color=ColorSchemes.ENERGY_COLORS["demand"],
            alpha=0.8,
        )

        # Net peak bars (after PV)
        if monthly_net_peak is not None:
            ax.bar(
                months + width/2,
                monthly_net_peak.reindex(months).fillna(0).clip(lower=0),
                width,
                label="Net Peak (with PV)",
                color=ColorSchemes.ENERGY_COLORS["self_consumed"],
                alpha=0.8,
            )

        self.format_axis(
            ax,
            xlabel="Month",
            ylabel="Peak Power (kW)",
            legend=True,
            legend_loc="upper right",
        )

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - {weather}\nMonthly Peak Demand",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig
