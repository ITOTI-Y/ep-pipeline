"""
Base figure class for visualization.

Provides common functionality for all figure types:
- Matplotlib configuration
- Consistent styling
- Multi-format export
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ..config import FigureConfig


class BaseFigure(ABC):
    """
    Abstract base class for all figures.

    Provides consistent styling and export functionality
    compliant with Building and Environment journal standards.
    """

    def __init__(self, config: FigureConfig | None = None):
        """
        Initialize base figure.

        Args:
            config: Figure configuration. Uses default if not provided.
        """
        self.config = config or FigureConfig()
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """Configure matplotlib with journal-compliant settings."""
        plt.rcParams.update(self.config.get_rc_params())

    @abstractmethod
    def plot(self, data: Any, **kwargs) -> plt.Figure:
        """
        Generate the figure.

        Args:
            data: Input data (type depends on figure)
            **kwargs: Additional arguments

        Returns:
            matplotlib Figure object
        """
        pass

    def save(
        self,
        fig: plt.Figure,
        filename: str,
        output_dir: Path | str,
        formats: list[str] | None = None,
        dpi: int | None = None,
    ) -> list[Path]:
        """
        Save figure in multiple formats.

        Args:
            fig: matplotlib Figure to save
            filename: Base filename (without extension)
            output_dir: Output directory path
            formats: List of formats to save. Uses config default if None.
            dpi: DPI for raster formats. Uses config default if None.

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        formats = formats or self.config.output_formats
        dpi = dpi or self.config.dpi_combination

        saved_paths = []

        for fmt in formats:
            filepath = output_dir / f"{filename}.{fmt}"

            # Use higher DPI for certain formats
            save_dpi = self._get_dpi_for_format(fmt, dpi)

            fig.savefig(
                filepath,
                format=fmt,
                dpi=save_dpi,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor="white",
                edgecolor="none",
            )
            saved_paths.append(filepath)

        return saved_paths

    def _get_dpi_for_format(self, fmt: str, base_dpi: int) -> int:
        """
        Get appropriate DPI for a format.

        Args:
            fmt: File format
            base_dpi: Base DPI setting

        Returns:
            DPI to use for this format
        """
        if fmt in ["pdf", "eps", "svg"]:
            # Vector formats - DPI only affects rasterized elements
            return self.config.dpi_line_art
        elif fmt == "tiff":
            return self.config.dpi_combination
        elif fmt == "png":
            return base_dpi
        else:
            return base_dpi

    def create_figure(
        self,
        width: str = "double",
        height: float | None = None,
        nrows: int = 1,
        ncols: int = 1,
        **kwargs,
    ) -> tuple[plt.Figure, Any]:
        """
        Create a figure with standard sizing.

        Args:
            width: "single" or "double" column width
            height: Height in inches. Auto-calculated if None.
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            **kwargs: Additional arguments for plt.subplots

        Returns:
            Tuple of (Figure, Axes)
        """
        # Determine width
        if width == "single":
            fig_width = self.config.single_column_width
        else:
            fig_width = self.config.double_column_width

        # Determine height
        if height is None:
            if width == "single":
                fig_height = self.config.single_column_height * nrows
            else:
                fig_height = self.config.double_column_height * nrows
        else:
            fig_height = height

        # Create figure
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_width, fig_height),
            **kwargs,
        )

        return fig, axes

    def add_panel_label(
        self,
        ax: plt.Axes,
        label: str,
        loc: str = "upper left",
        fontweight: str = "bold",
    ) -> None:
        """
        Add panel label (a), (b), etc. to subplot.

        Args:
            ax: Axes to add label to
            label: Label text (e.g., "(a)")
            loc: Location of label
            fontweight: Font weight for label
        """
        # Position mapping
        positions = {
            "upper left": (0.02, 0.98),
            "upper right": (0.98, 0.98),
            "lower left": (0.02, 0.02),
            "lower right": (0.98, 0.02),
        }

        x, y = positions.get(loc, (0.02, 0.98))
        ha = "left" if "left" in loc else "right"
        va = "top" if "upper" in loc else "bottom"

        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=self.config.font_size_title,
            fontweight=fontweight,
            ha=ha,
            va=va,
        )

    def format_axis(
        self,
        ax: plt.Axes,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        legend: bool = True,
        legend_loc: str = "best",
        grid: bool = False,
    ) -> None:
        """
        Apply standard formatting to an axis.

        Args:
            ax: Axes to format
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Subplot title
            xlim: X-axis limits
            ylim: Y-axis limits
            legend: Whether to show legend
            legend_loc: Legend location
            grid: Whether to show grid
        """
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=self.config.font_size_title)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc=legend_loc, frameon=False)

        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # Ensure spines are thin
        for spine in ax.spines.values():
            spine.set_linewidth(self.config.line_width_thin)

    def close(self, fig: plt.Figure) -> None:
        """
        Close a figure to free memory.

        Args:
            fig: Figure to close
        """
        plt.close(fig)
