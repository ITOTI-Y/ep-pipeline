from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import ultraplot as uplt

from backend.utils.config import ConfigManager
from backend.viz.style import (
    BUILDING_AND_ENVIRONMENT_STYLE,
    FigureWidth,
    ImageType,
)


class ChartGenerator:
    """Chart generator for PV simulation results visualization."""

    def __init__(
        self,
    ):
        self.config = ConfigManager(Path("backend/configs"))
        self.paths = self.config.paths
        self.output_dir = self.paths.visualization_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = BUILDING_AND_ENVIRONMENT_STYLE
        uplt.rc.update(self.style.get_rc_params())

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
        plt.close(fig)
        return path
