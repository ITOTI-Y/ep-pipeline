"""
Storage-related figures (F1-F3).

F1: Storage SOC hourly curve with 100% line and curtailment zones
F2: Typical day storage operation (summer/winter)
F3: Monthly curtailment statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseFigure
from ..config import (
    FigureConfig,
    ColorSchemes,
    STORAGE_CAPACITY,
    BUILDINGS_WITH_STORAGE,
)


class StorageSOCFigure(BaseFigure):
    """
    F1: Storage battery hourly State of Charge curve.

    Shows:
    - Hourly SOC throughout the year
    - 100% full capacity reference line
    - Curtailment periods highlighted
    """

    def plot(
        self,
        df: pd.DataFrame,
        building: str,
        weather: str,
        curtailment_series: pd.Series | None = None,
    ) -> plt.Figure:
        """
        Plot storage SOC curve.

        Args:
            df: Processed hourly DataFrame with storage_soc_pct column
            building: Building type
            weather: Weather scenario
            curtailment_series: Optional hourly curtailment data

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=2.5)

        capacity_kwh = STORAGE_CAPACITY.get(building, 0)

        if capacity_kwh == 0 or "storage_soc_pct" not in df.columns:
            ax.text(
                0.5, 0.5,
                f"No storage system for {building}",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=self.config.font_size_normal,
            )
            return fig

        # Get SOC data
        soc = df["storage_soc_pct"].values
        hours = np.arange(len(soc))

        # Plot SOC curve
        ax.plot(
            hours, soc,
            color=ColorSchemes.ENERGY_COLORS["storage_charge"],
            linewidth=0.3,
            alpha=0.8,
            label="State of Charge",
        )

        # 100% full capacity line
        ax.axhline(
            y=100,
            color=ColorSchemes.ENERGY_COLORS["self_consumed"],
            linestyle="--",
            linewidth=1.0,
            label="Full capacity (100%)",
        )

        # 50% reference line
        ax.axhline(
            y=50,
            color="#aaaaaa",
            linestyle=":",
            linewidth=0.5,
            alpha=0.5,
        )

        # Highlight curtailment periods
        # if curtailment_series is not None and len(curtailment_series) == len(soc):
        #     curtailment_mask = curtailment_series.values > 0
        #     if curtailment_mask.any():
        #         # Fill curtailment periods at top
        #         ax.fill_between(
        #             hours,
        #             soc,
        #             100,
        #             where=curtailment_mask & (soc >= 95),
        #             color=ColorSchemes.ENERGY_COLORS["curtailed"],
        #             alpha=0.5,
        #             label="Curtailment period",
        #         )

        # Format
        self.format_axis(
            ax,
            xlabel="Hour of year",
            ylabel="State of Charge (%)",
            xlim=(0, 8760),
            ylim=(0, 105),
            legend=False,  # Disable auto legend
        )

        # Add legend outside the plot area to avoid overlap
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.15),
            ncol=3,
            frameon=False,
            fontsize=self.config.font_size_subscript,
        )

        # Add title with building and weather info
        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(f"{building_name} - {weather}", fontsize=self.config.font_size_title)

        # Add capacity annotation
        ax.annotate(
            f"Capacity: {capacity_kwh} kWh",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=self.config.font_size_subscript,
            ha="left",
            va="top",
        )

        fig.tight_layout()
        return fig


class TypicalDayStorageFigure(BaseFigure):
    """
    F2: Typical day storage operation.

    Shows summer and winter typical day comparison:
    - PV generation profile
    - Building demand profile
    - Storage charge/discharge
    """

    def plot(
        self,
        summer_df: pd.DataFrame,
        winter_df: pd.DataFrame,
        building: str,
    ) -> plt.Figure:
        """
        Plot typical day storage operation.

        Args:
            summer_df: Summer day (24 hours) DataFrame
            winter_df: Winter day (24 hours) DataFrame
            building: Building type

        Returns:
            matplotlib Figure
        """
        fig, axes = self.create_figure(
            width="double",
            height=2.5,
            ncols=2,
            sharey=True,
        )

        capacity_kwh = STORAGE_CAPACITY.get(building, 0)
        building_name = ColorSchemes.get_building_name(building)

        for ax, day_df, season in zip(
            axes,
            [summer_df, winter_df],
            ["Summer (Jul 15)", "Winter (Jan 15)"],
        ):
            self._plot_single_day(ax, day_df, season, capacity_kwh)

        # Common legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            fontsize=self.config.font_size_subscript,
            bbox_to_anchor=(0.5, 1.02),
        )

        # Remove individual legends
        for ax in axes:
            legend = ax.get_legend()
            if legend:
                legend.remove()

        fig.suptitle(
            f"{building_name} - Typical Day Operation",
            fontsize=self.config.font_size_title,
            y=1.08,
        )

        fig.tight_layout()
        return fig

    def _plot_single_day(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        title: str,
        capacity_kwh: float,
    ) -> None:
        """Plot single day data."""
        hours = np.arange(24)

        # Ensure we have 24 hours of data
        if len(df) != 24:
            # Try to get hourly mean if data is different length
            if "hour" in df.columns:
                df = df.groupby("hour").mean()
            else:
                df = df.head(24)

        # PV Generation (area)
        if "pv_generation_kw" in df.columns:
            pv = df["pv_generation_kw"].values[:24]
            ax.fill_between(
                hours, 0, pv,
                alpha=0.6,
                color=ColorSchemes.ENERGY_COLORS["pv_generation"],
                label="PV Generation",
            )

        # Demand (line)
        if "demand_kw" in df.columns:
            demand = df["demand_kw"].values[:24]
            ax.plot(
                hours, demand,
                color=ColorSchemes.ENERGY_COLORS["demand"],
                linewidth=1.5,
                label="Building Demand",
            )

        # Storage charge (positive bars)
        if "storage_charge_power_kw" in df.columns:
            charge = df["storage_charge_power_kw"].values[:24]
            ax.bar(
                hours, charge,
                width=0.6,
                alpha=0.7,
                color=ColorSchemes.ENERGY_COLORS["storage_charge"],
                label="Storage Charging",
            )

        # Storage discharge (negative bars)
        if "storage_discharge_power_kw" in df.columns:
            discharge = df["storage_discharge_power_kw"].values[:24]
            ax.bar(
                hours, -discharge,
                width=0.6,
                alpha=0.7,
                color=ColorSchemes.ENERGY_COLORS["storage_discharge"],
                label="Storage Discharging",
            )

        self.format_axis(
            ax,
            xlabel="Hour",
            ylabel="Power (kW)",
            title=title,
            xlim=(-0.5, 23.5),
        )

        ax.set_xticks([0, 6, 12, 18, 24])
        ax.axhline(y=0, color="black", linewidth=0.5)


class MonthlyCurtailmentFigure(BaseFigure):
    """
    F3: Monthly curtailment statistics.

    Shows:
    - Monthly curtailed energy
    - Curtailment rate trend
    """

    def plot(
        self,
        monthly_curtailment: pd.Series,
        monthly_pv: pd.Series,
        building: str,
        weather: str,
    ) -> plt.Figure:
        """
        Plot monthly curtailment statistics.

        Args:
            monthly_curtailment: Monthly curtailed energy (kWh)
            monthly_pv: Monthly PV generation (kWh)
            building: Building type
            weather: Weather scenario

        Returns:
            matplotlib Figure
        """
        fig, ax1 = self.create_figure(width="single", height=2.5)

        months = np.arange(1, 13)
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

        # Ensure data is aligned
        curtailment = monthly_curtailment.reindex(months).fillna(0).values
        pv_gen = monthly_pv.reindex(months).fillna(0).values

        # Calculate curtailment rate
        curtailment_rate = np.where(pv_gen > 0, curtailment / pv_gen * 100, 0)

        # Bar chart for curtailed energy
        bars = ax1.bar(
            months,
            curtailment,
            color=ColorSchemes.ENERGY_COLORS["curtailed"],
            alpha=0.8,
            label="Curtailed Energy",
            width=0.7,
        )

        ax1.set_xlabel("Month")
        ax1.set_ylabel("Curtailed Energy (kWh)")
        ax1.set_xticks(months)
        ax1.set_xticklabels(month_labels)

        # Secondary axis for curtailment rate
        ax2 = ax1.twinx()
        ax2.plot(
            months,
            curtailment_rate,
            color=ColorSchemes.ENERGY_COLORS["grid_import"],
            marker="o",
            markersize=3,
            linewidth=1.0,
            label="Curtailment Rate",
        )
        ax2.set_ylabel("Curtailment Rate (%)")
        ax2.set_ylim(0, max(curtailment_rate) * 1.2 if max(curtailment_rate) > 0 else 10)

        # Combined legend - moved outside to avoid overlap
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.18),
            ncol=2,
            frameon=False,
            fontsize=self.config.font_size_subscript,
        )

        # Title
        building_name = ColorSchemes.get_building_name(building)
        ax1.set_title(
            f"{building_name} - {weather}",
            fontsize=self.config.font_size_title,
        )

        # Add total annotation
        total_curtailed = curtailment.sum()
        avg_rate = np.mean(curtailment_rate[pv_gen > 0]) if (pv_gen > 0).any() else 0
        ax1.annotate(
            f"Total: {total_curtailed:.0f} kWh\nAvg Rate: {avg_rate:.1f}%",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            fontsize=self.config.font_size_subscript,
            ha="right",
            va="top",
        )

        fig.tight_layout()
        return fig


class StorageComparisonFigure(BaseFigure):
    """
    Comparison of storage performance across climate scenarios.

    Supplementary figure for residential buildings.
    """

    def plot(
        self,
        metrics_dict: dict[str, dict],
        building: str,
    ) -> plt.Figure:
        """
        Plot storage metrics comparison.

        Args:
            metrics_dict: Dict mapping weather to storage metrics
            building: Building type

        Returns:
            matplotlib Figure
        """
        fig, axes = self.create_figure(
            width="double",
            height=2.0,
            ncols=3,
        )

        weathers = list(metrics_dict.keys())
        colors = [ColorSchemes.get_climate_color(w) for w in weathers]

        # Metrics to plot
        metric_configs = [
            ("total_discharged_kwh", "Discharged Energy (kWh)"),
            ("full_cycles", "Full Equivalent Cycles"),
            ("utilization_rate", "Utilization Rate (%)"),
        ]

        for ax, (metric, ylabel) in zip(axes, metric_configs):
            values = [metrics_dict[w].get(metric, 0) for w in weathers]

            bars = ax.bar(
                range(len(weathers)),
                values,
                color=colors,
                alpha=0.8,
            )

            ax.set_xticks(range(len(weathers)))
            ax.set_xticklabels(weathers, rotation=45, ha="right")
            ax.set_ylabel(ylabel)

        building_name = ColorSchemes.get_building_name(building)
        fig.suptitle(
            f"{building_name} - Storage Performance by Climate",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig
