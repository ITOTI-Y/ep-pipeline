"""
ECM sensitivity analysis figures (F7-F8).

F7: ECM parameter sensitivity heatmap
F8: Sensitivity index bar chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from .base import BaseFigure
from ..config import (
    FigureConfig,
    ColorSchemes,
    ECM_PARAMETERS,
    EUI_TARGETS,
    ALL_BUILDINGS,
)


class ECMSensitivityHeatmap(BaseFigure):
    """
    F7: ECM parameter sensitivity heatmap.

    Shows Spearman correlation coefficients between
    ECM parameters and EUI targets.
    """

    def plot(
        self,
        ecm_df: pd.DataFrame,
        building_filter: str | None = None,
    ) -> plt.Figure:
        """
        Plot sensitivity heatmap.

        Args:
            ecm_df: ECM results DataFrame with parameters and EUI values
            building_filter: Optional building type to filter

        Returns:
            matplotlib Figure
        """
        fig, axes = self.create_figure(
            width="double",
            height=4.5,  # Increased height to avoid overlap
            ncols=3,
        )

        # Filter by building if specified
        if building_filter and "building_type" in ecm_df.columns:
            df = ecm_df[ecm_df["building_type"] == building_filter].copy()
        else:
            df = ecm_df.copy()

        # Get available ECM parameters
        available_params = [p for p in ECM_PARAMETERS if p in df.columns]
        available_targets = [t for t in EUI_TARGETS if t in df.columns]

        if not available_params or not available_targets:
            for ax in axes:
                ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        # Calculate correlations for each target
        eui_labels = {
            "total_site_eui": "Total Site EUI",
            "net_site_eui": "Net Site EUI",
            "total_source_eui": "Total Source EUI",
        }

        for ax, target in zip(axes, available_targets):
            # Calculate Spearman correlation
            correlations = df[available_params].corrwith(
                df[target], method="spearman"
            )

            # Create matrix for heatmap (1 row)
            corr_matrix = correlations.values.reshape(1, -1)

            # Plot heatmap
            im = ax.imshow(
                corr_matrix,
                cmap=ColorSchemes.DIVERGING_CMAP,
                vmin=-1,
                vmax=1,
                aspect="auto",
            )

            # Labels
            ax.set_xticks(range(len(available_params)))
            ax.set_xticklabels(
                [ColorSchemes.get_ecm_name(p) for p in available_params],
                rotation=45,
                ha="right",
                fontsize=self.config.font_size_subscript,
            )
            ax.set_yticks([])
            ax.set_title(
                eui_labels.get(target, target),
                fontsize=self.config.font_size_title,
            )

            # Add correlation values as text
            for i, corr in enumerate(correlations):
                text_color = "white" if abs(corr) > 0.5 else "black"
                ax.text(
                    i, 0, f"{corr:.2f}",
                    ha="center", va="center",
                    fontsize=self.config.font_size_subscript,
                    color=text_color,
                )

        # Colorbar
        cbar = fig.colorbar(
            im, ax=axes, shrink=0.6, pad=0.02,
            label="Spearman Correlation",
        )
        cbar.ax.tick_params(labelsize=self.config.font_size_subscript)

        # Title
        title = "ECM Parameter Sensitivity Analysis"
        if building_filter:
            title += f" - {ColorSchemes.get_building_name(building_filter)}"
        fig.suptitle(title, fontsize=self.config.font_size_title, y=1.02)

        fig.tight_layout()
        return fig

    def plot_by_building(
        self,
        ecm_df: pd.DataFrame,
        target: str = "total_site_eui",
    ) -> plt.Figure:
        """
        Plot sensitivity heatmap for all buildings.

        Args:
            ecm_df: ECM results DataFrame
            target: EUI target to analyze

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=5.0)  # Increased height

        # Get available parameters and buildings
        available_params = [p for p in ECM_PARAMETERS if p in ecm_df.columns]
        buildings = ecm_df["building_type"].unique() if "building_type" in ecm_df.columns else []

        if not available_params or len(buildings) == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        # Calculate correlations for each building
        corr_matrix = []
        building_labels = []

        for building in buildings:
            df_building = ecm_df[ecm_df["building_type"] == building]
            if len(df_building) > 10 and target in df_building.columns:
                correlations = df_building[available_params].corrwith(
                    df_building[target], method="spearman"
                )
                corr_matrix.append(correlations.values)
                building_labels.append(ColorSchemes.get_building_name(building))

        if not corr_matrix:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        corr_matrix = np.array(corr_matrix)

        # Heatmap
        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap=ColorSchemes.DIVERGING_CMAP,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": self.config.font_size_subscript},
            xticklabels=[ColorSchemes.get_ecm_name(p) for p in available_params],
            yticklabels=building_labels,
            cbar_kws={"shrink": 0.8, "label": "Spearman Correlation"},
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

        eui_labels = {
            "total_site_eui": "Total Site EUI",
            "net_site_eui": "Net Site EUI",
            "total_source_eui": "Total Source EUI",
        }
        ax.set_title(
            f"ECM Sensitivity by Building Type - {eui_labels.get(target, target)}",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig


class SensitivityIndexFigure(BaseFigure):
    """
    F8: ECM parameter sensitivity index bar chart.

    Uses standardized regression coefficients as sensitivity indices.
    """

    def plot(
        self,
        ecm_df: pd.DataFrame,
        building: str,
        target: str = "total_site_eui",
    ) -> plt.Figure:
        """
        Plot sensitivity index bar chart.

        Args:
            ecm_df: ECM results DataFrame
            building: Building type to analyze
            target: EUI target

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="single", height=3.5)

        # Filter by building
        if "building_type" in ecm_df.columns:
            df = ecm_df[ecm_df["building_type"] == building].copy()
        else:
            df = ecm_df.copy()

        # Get available parameters
        available_params = [p for p in ECM_PARAMETERS if p in df.columns]

        if not available_params or target not in df.columns or len(df) < 20:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        # Calculate standardized regression coefficients
        X = df[available_params].values
        y = df[target].values

        # Handle missing values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        if len(X) < 20:
            ax.text(0.5, 0.5, "Insufficient data after filtering", ha="center", va="center")
            return fig

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Get coefficients (absolute values for importance)
        coefficients = model.coef_
        importance = np.abs(coefficients)
        importance_normalized = importance / importance.sum() * 100

        # Sort by importance
        sorted_idx = np.argsort(importance_normalized)[::-1]
        sorted_params = [available_params[i] for i in sorted_idx]
        sorted_importance = importance_normalized[sorted_idx]
        sorted_coef = coefficients[sorted_idx]

        # Colors based on positive/negative effect
        colors = [
            ColorSchemes.ENERGY_COLORS["grid_import"] if c > 0
            else ColorSchemes.ENERGY_COLORS["self_consumed"]
            for c in sorted_coef
        ]

        # Horizontal bar chart
        y_pos = np.arange(len(sorted_params))
        bars = ax.barh(
            y_pos,
            sorted_importance,
            color=colors,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [ColorSchemes.get_ecm_name(p) for p in sorted_params],
            fontsize=self.config.font_size_subscript,
        )
        ax.set_xlabel("Relative Importance (%)")
        ax.invert_yaxis()

        # Extend xlim to accommodate labels
        max_importance = max(sorted_importance) if len(sorted_importance) > 0 else 100
        ax.set_xlim(0, max_importance * 1.25)  # Add 25% margin for labels

        # Add value labels
        for bar, val, coef in zip(bars, sorted_importance, sorted_coef):
            sign = "+" if coef > 0 else "-"
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.1f}%",
                va="center",
                fontsize=self.config.font_size_subscript,
            )

        # Legend for color meaning
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ColorSchemes.ENERGY_COLORS["grid_import"],
                  alpha=0.8, label="Increases EUI"),
            Patch(facecolor=ColorSchemes.ENERGY_COLORS["self_consumed"],
                  alpha=0.8, label="Decreases EUI"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            frameon=False,
            fontsize=self.config.font_size_subscript,
        )

        building_name = ColorSchemes.get_building_name(building)
        eui_labels = {
            "total_site_eui": "Total Site EUI",
            "net_site_eui": "Net Site EUI",
            "total_source_eui": "Total Source EUI",
        }
        ax.set_title(
            f"{building_name}\n{eui_labels.get(target, target)} Sensitivity",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig

    def plot_comparison(
        self,
        ecm_df: pd.DataFrame,
        target: str = "total_site_eui",
    ) -> plt.Figure:
        """
        Plot sensitivity comparison across all buildings.

        Args:
            ecm_df: ECM results DataFrame
            target: EUI target

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure(width="double", height=4.0)

        buildings = ecm_df["building_type"].unique() if "building_type" in ecm_df.columns else []
        available_params = [p for p in ECM_PARAMETERS if p in ecm_df.columns]

        if not available_params or len(buildings) == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        # Calculate importance for each building
        importance_matrix = []
        building_labels = []

        for building in buildings:
            df = ecm_df[ecm_df["building_type"] == building]

            if len(df) < 20 or target not in df.columns:
                continue

            X = df[available_params].values
            y = df[target].values

            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]

            if len(X) < 20:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            importance = np.abs(model.coef_)
            importance_normalized = importance / importance.sum() * 100

            importance_matrix.append(importance_normalized)
            building_labels.append(ColorSchemes.get_building_name(building))

        if not importance_matrix:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        importance_matrix = np.array(importance_matrix)

        # Heatmap
        sns.heatmap(
            importance_matrix,
            ax=ax,
            cmap=ColorSchemes.SEQUENTIAL_CMAP,
            annot=True,
            fmt=".1f",
            annot_kws={"size": self.config.font_size_subscript},
            xticklabels=[ColorSchemes.get_ecm_name(p) for p in available_params],
            yticklabels=building_labels,
            cbar_kws={"shrink": 0.8, "label": "Relative Importance (%)"},
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=self.config.font_size_subscript,
        )

        eui_labels = {
            "total_site_eui": "Total Site EUI",
            "net_site_eui": "Net Site EUI",
            "total_source_eui": "Total Source EUI",
        }
        ax.set_title(
            f"ECM Parameter Importance - {eui_labels.get(target, target)}",
            fontsize=self.config.font_size_title,
        )

        fig.tight_layout()
        return fig
