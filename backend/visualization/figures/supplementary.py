"""
Supplementary research figures (S1-S8).

S1: Energy breakdown comparison
S2: ECM parameter correlation matrix
S3: Climate resilience radar chart
S4: PV efficiency vs temperature
S5: Energy flow Sankey diagram
S6: Load curve clustering
S7: Carbon reduction comparison
S8: Comprehensive performance radar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from .base import BaseFigure
from ..config import (
    FigureConfig,
    ColorSchemes,
    ALL_CLIMATES,
    ALL_BUILDINGS,
    ECM_PARAMETERS,
    GRID_EMISSION_FACTOR,
)


class EnergyBreakdownFigure(BaseFigure):
    """
    S1: Energy breakdown comparison (before/after optimization).
    """

    def plot(
        self,
        baseline_breakdown: dict[str, float],
        optimized_breakdown: dict[str, float],
        building: str,
    ) -> plt.Figure:
        """
        Plot energy breakdown stacked bar comparison.

        Args:
            baseline_breakdown: {end_use: energy_kwh} for baseline
            optimized_breakdown: {end_use: energy_kwh} for optimized
            building: Building type

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=3.0)

        categories = list(baseline_breakdown.keys())
        baseline_vals = [baseline_breakdown.get(c, 0) for c in categories]
        optimized_vals = [optimized_breakdown.get(c, 0) for c in categories]

        x = np.arange(2)
        width = 0.6

        # Color palette
        colors = plt.colormaps["Set2"](np.linspace(0, 1, len(categories)))

        # Stacked bars
        bottom_baseline = np.zeros(1)
        bottom_optimized = np.zeros(1)

        for i, (cat, color) in enumerate(zip(categories, colors)):
            ax.bar(0, baseline_vals[i], width, bottom=bottom_baseline[0],
                   label=cat, color=color, alpha=0.8)
            ax.bar(1, optimized_vals[i], width, bottom=bottom_optimized[0],
                   color=color, alpha=0.8)
            bottom_baseline[0] += baseline_vals[i]
            bottom_optimized[0] += optimized_vals[i]

        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Optimized"])
        ax.set_ylabel("Energy (kWh)")
        ax.legend(
            loc="upper right",
            fontsize=self.config.font_size_subscript,
            frameon=False,
        )

        # Reduction annotation
        total_baseline = sum(baseline_vals)
        total_optimized = sum(optimized_vals)
        reduction = (total_baseline - total_optimized) / total_baseline * 100

        ax.annotate(
            f"Total Reduction: {reduction:.1f}%",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            fontsize=self.config.font_size_subscript,
        )

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - Energy Breakdown",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class ECMCorrelationMatrix(BaseFigure):
    """
    S2: ECM parameter correlation matrix.
    """

    def plot(
        self,
        ecm_df: pd.DataFrame,
    ) -> plt.Figure:
        """
        Plot ECM parameter correlation matrix.

        Args:
            ecm_df: ECM results DataFrame

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=5.0)

        # Get available parameters
        available_params = [p for p in ECM_PARAMETERS if p in ecm_df.columns]

        if not available_params:
            ax.text(0.5, 0.5, "No ECM parameters found", ha="center", va="center")
            return fig

        # Calculate correlation matrix
        corr_matrix = ecm_df[available_params].corr()

        # Rename for display
        display_names = [ColorSchemes.get_ecm_name(p) for p in available_params]
        corr_matrix.index = display_names
        corr_matrix.columns = display_names

        # Heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            ax=ax,
            mask=mask,
            cmap=ColorSchemes.DIVERGING_CMAP,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": self.config.font_size_subscript},
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=self.config.font_size_subscript,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=self.config.font_size_subscript,
        )

        ax.set_title(
            "ECM Parameter Correlation Matrix",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class ClimateResilienceRadar(BaseFigure):
    """
    S3: Climate resilience radar chart.
    """

    def plot(
        self,
        metrics_by_climate: dict[str, dict[str, float]],
        building: str,
    ) -> plt.Figure:
        """
        Plot climate resilience radar chart.

        Args:
            metrics_by_climate: {weather: {metric: value}}
            building: Building type

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=3.5)
        ax.remove()
        ax = fig.add_subplot(111, projection="polar")

        metrics = ["EUI Change", "Peak Change", "PV Output", "Comfort Hours", "Cooling Load"]
        n_metrics = len(metrics)

        # Angles for radar
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        for weather, data in metrics_by_climate.items():
            values = [data.get(m, 0) for m in metrics]
            values += values[:1]

            ax.plot(
                angles,
                values,
                linewidth=1.5,
                color=ColorSchemes.get_climate_color(weather),
                label=weather,
            )
            ax.fill(
                angles,
                values,
                alpha=0.1,
                color=ColorSchemes.get_climate_color(weather),
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=self.config.font_size_subscript)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.0),
            fontsize=self.config.font_size_subscript,
            frameon=False,
        )

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name}\nClimate Resilience",
            fontsize=self.config.font_size_title,
            pad=20,
        )

        fig.tight_layout()
        return fig


class PVEfficiencyTempFigure(BaseFigure):
    """
    S4: PV system efficiency vs temperature.
    """

    def plot(
        self,
        df: pd.DataFrame,
        building: str,
        weather: str,
    ) -> plt.Figure:
        """
        Plot PV efficiency vs outdoor temperature.

        Args:
            df: Processed hourly DataFrame
            building: Building type
            weather: Weather scenario

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=2.5)

        if "pv_generation_kw" not in df.columns or "outdoor_temp_c" not in df.columns:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        # Filter hours with PV generation
        mask = df["pv_generation_kw"] > 0.1
        temp = df.loc[mask, "outdoor_temp_c"]
        pv = df.loc[mask, "pv_generation_kw"]

        # Normalize PV to peak (as proxy for efficiency)
        pv_normalized = pv / pv.max() * 100

        # Scatter plot with density coloring
        ax.scatter(
            temp,
            pv_normalized,
            c=ColorSchemes.ENERGY_COLORS["pv_generation"],
            alpha=0.3,
            s=5,
            edgecolors="none",
        )

        # Add trend line
        if len(temp) > 10:
            z = np.polyfit(temp, pv_normalized, 1)
            p = np.poly1d(z)
            temp_range = np.linspace(temp.min(), temp.max(), 100)
            ax.plot(
                temp_range,
                p(temp_range),
                color="red",
                linewidth=1.5,
                label=f"Trend (slope: {z[0]:.2f}%/°C)",
            )

        self.format_axis(
            ax,
            xlabel="Outdoor Temperature (°C)",
            ylabel="Normalized PV Output (%)",
            legend=True,
            legend_loc="upper right",
        )

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - {weather}\nPV Performance vs Temperature",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class EnergySankeyFigure(BaseFigure):
    """
    S5: Energy flow Sankey diagram (simplified version).
    """

    def plot(
        self,
        energy_flows: dict[str, float],
        building: str,
        weather: str,
    ) -> plt.Figure:
        """
        Plot simplified energy flow diagram.

        Args:
            energy_flows: {flow_name: energy_kwh}
            building: Building type
            weather: Weather scenario

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=3.0)

        # Extract flows
        pv_gen = energy_flows.get("pv_generation", 0)
        self_consumed = energy_flows.get("self_consumed", 0)
        grid_export = energy_flows.get("grid_export", 0)
        grid_import = energy_flows.get("grid_import", 0)
        storage_throughput = energy_flows.get("storage_throughput", 0)
        demand = energy_flows.get("demand", 0)

        # Simplified horizontal bar representation
        y_positions = [4, 3, 2, 1, 0]
        labels = ["PV Generation", "Self-Consumed", "Grid Export", "Grid Import", "Total Demand"]
        values = [pv_gen, self_consumed, grid_export, grid_import, demand]
        colors = [
            ColorSchemes.ENERGY_COLORS["pv_generation"],
            ColorSchemes.ENERGY_COLORS["self_consumed"],
            ColorSchemes.ENERGY_COLORS["grid_export"],
            ColorSchemes.ENERGY_COLORS["grid_import"],
            ColorSchemes.ENERGY_COLORS["demand"],
        ]

        # Normalize to max for visualization
        max_val = max(values) if max(values) > 0 else 1
        normalized = [v / max_val for v in values]

        bars = ax.barh(
            y_positions,
            normalized,
            height=0.6,
            color=colors,
            alpha=0.8,
            edgecolor="white",
        )

        # Add value labels
        for y, val, norm in zip(y_positions, values, normalized):
            ax.text(
                norm + 0.02,
                y,
                f"{val/1000:.1f} MWh",
                va="center",
                fontsize=self.config.font_size_subscript,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.3)
        ax.set_xlabel("Relative Energy Flow")
        ax.set_xticks([])

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name} - {weather}\nAnnual Energy Flow",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class LoadClusteringFigure(BaseFigure):
    """
    S6: Load curve clustering analysis.
    """

    def plot(
        self,
        df: pd.DataFrame,
        building: str,
        n_clusters: int = 4,
    ) -> plt.Figure:
        """
        Plot typical load patterns from clustering.

        Args:
            df: Processed hourly DataFrame
            building: Building type
            n_clusters: Number of clusters

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=2.5)

        if "demand_kw" not in df.columns or "hour" not in df.columns:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        # Reshape to daily profiles
        try:
            daily_profiles = df.pivot_table(
                index=df.index.date,
                columns="hour",
                values="demand_kw",
                aggfunc="mean",
            ).dropna()
        except Exception:
            # Fallback: group by dayofyear and hour
            daily_profiles = df.groupby(["dayofyear", "hour"])["demand_kw"].mean().unstack()
            daily_profiles = daily_profiles.dropna()

        if len(daily_profiles) < n_clusters:
            ax.text(0.5, 0.5, "Insufficient data for clustering", ha="center", va="center")
            return fig

        # Simple clustering using k-means
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(daily_profiles.values)

        # Plot cluster centroids
        hours = np.arange(24)
        colors = plt.colormaps["Set1"](np.linspace(0, 1, n_clusters))

        for i in range(n_clusters):
            centroid = kmeans.cluster_centers_[i]
            count = (clusters == i).sum()
            ax.plot(
                hours,
                centroid,
                color=colors[i],
                linewidth=1.5,
                label=f"Pattern {i+1} (n={count})",
            )

        self.format_axis(
            ax,
            xlabel="Hour",
            ylabel="Demand (kW)",
            xlim=(-0.5, 23.5),
            legend=True,
            legend_loc="upper right",
        )

        ax.set_xticks([0, 6, 12, 18, 24])

        building_name = ColorSchemes.get_building_name(building)
        ax.set_title(
            f"{building_name}\nTypical Daily Load Patterns",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class CarbonReductionFigure(BaseFigure):
    """
    S7: Carbon emission reduction comparison.
    """

    def plot(
        self,
        carbon_data: dict[str, dict[str, float]],
    ) -> plt.Figure:
        """
        Plot carbon reduction comparison.

        Args:
            carbon_data: {building: {baseline_kg, pv_kg, reduction_kg}}

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=3.0)

        buildings = list(carbon_data.keys())
        x = np.arange(len(buildings))
        width = 0.35

        baseline_vals = [carbon_data[b].get("baseline_kg", 0) / 1000 for b in buildings]
        pv_vals = [carbon_data[b].get("pv_kg", 0) / 1000 for b in buildings]

        ax.bar(
            x - width/2,
            baseline_vals,
            width,
            label="Baseline",
            color=ColorSchemes.ENERGY_COLORS["grid_import"],
            alpha=0.8,
        )
        ax.bar(
            x + width/2,
            pv_vals,
            width,
            label="With PV",
            color=ColorSchemes.ENERGY_COLORS["self_consumed"],
            alpha=0.8,
        )

        # Add reduction percentages
        for i, (base, pv) in enumerate(zip(baseline_vals, pv_vals)):
            if base > 0:
                reduction = (base - pv) / base * 100
                ax.annotate(
                    f"-{reduction:.0f}%",
                    xy=(i, max(base, pv)),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=self.config.font_size_subscript,
                    fontweight="bold",
                )

        self.format_axis(
            ax,
            xlabel="Building Type",
            ylabel="Annual CO₂ Emissions (tonnes)",
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
            "Carbon Emission Reduction from PV Integration",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class PerformanceRadarFigure(BaseFigure):
    """
    S8: Comprehensive performance radar chart.
    """

    def plot(
        self,
        performance_data: dict[str, dict[str, float]],
    ) -> plt.Figure:
        """
        Plot multi-dimensional performance radar.

        Args:
            performance_data: {building: {metric: normalized_value}}

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=4.0)
        ax.remove()
        ax = fig.add_subplot(111, projection="polar")

        metrics = [
            "EUI Reduction",
            "Self-Consumption",
            "Self-Sufficiency",
            "Peak Reduction",
            "Carbon Savings",
            "Cost Savings",
        ]
        n_metrics = len(metrics)

        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        for building, data in performance_data.items():
            values = [data.get(m, 0) for m in metrics]
            values += values[:1]

            ax.plot(
                angles,
                values,
                linewidth=2,
                color=ColorSchemes.get_building_color(building),
                label=ColorSchemes.get_building_name(building),
            )
            ax.fill(
                angles,
                values,
                alpha=0.15,
                color=ColorSchemes.get_building_color(building),
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=self.config.font_size_subscript)
        ax.set_ylim(0, 100)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.0),
            fontsize=self.config.font_size_subscript,
            frameon=False,
        )

        ax.set_title(
            "Comprehensive Performance Comparison",
            fontsize=self.config.font_size_title,
            pad=20,
        )

        fig.tight_layout()
        return fig
