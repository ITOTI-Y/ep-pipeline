"""Journal style configuration for Building and Environment."""

from dataclasses import dataclass, field
from enum import Enum


class FigureWidth(Enum):
    """Elsevier standard figure widths in inches."""

    SINGLE_COLUMN = 3.54  # 90 mm
    ONE_HALF_COLUMN = 5.51  # 140 mm
    DOUBLE_COLUMN = 7.48  # 190 mm


class ImageType(Enum):
    """Image type with corresponding DPI requirements."""

    LINE_ART = 300
    COMBINATION = 300
    HALFTONE = 300


@dataclass
class JournalStyle:
    """Building and Environment journal style configuration."""

    # Font settings
    font_family: str = "DejaVu Sans"
    font_size: float = 8.0
    font_size_small: float = 7.0
    font_size_title: float = 9.0
    font_size_label: float = 10.0
    font_size_tick_label: float = 8.0

    # Line settings
    line_width: float = 0.8
    line_width_thick: float = 1.2
    axis_line_width: float = 0.5
    tick_major_width: float = 0.5
    tick_minor_width: float = 0.3

    # Color-blind friendly palette (Wong)
    colors: tuple[str, ...] = field(
        default_factory=lambda: (
            "#0072B2",  # Blue
            "#E69F00",  # Orange
            "#009E73",  # Green
            "#CC79A7",  # Pink
            "#D55E00",  # Red-orange
            "#F0E442",  # Yellow
            "#56B4E9",  # Light Blue
            "#B2182B",  # Red
        )
    )

    # Default output settings
    default_width: float = 3.54
    default_dpi: int = 1000
    default_format: str = "png"

    show_grid: bool = False

    def get_rc_params(self) -> dict:
        """Return matplotlib/ultraplot rc parameters."""
        return {
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.linewidth": self.axis_line_width,
            "axes.labelsize": self.font_size_label,
            "axes.titlesize": self.font_size_title,
            "xtick.labelsize": self.font_size_tick_label,
            "ytick.labelsize": self.font_size_tick_label,
            "xtick.major.width": self.tick_major_width,
            "ytick.major.width": self.tick_major_width,
            "xtick.minor.width": self.tick_minor_width,
            "ytick.minor.width": self.tick_minor_width,
            "lines.linewidth": self.line_width,
            "legend.fontsize": self.font_size_small,
            "legend.frameon": False,
            "savefig.dpi": self.default_dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.grid": self.show_grid,
        }

    def get_color(self, index: int) -> str:
        """Return color at given index with cycling."""
        return self.colors[index % len(self.colors)]


BUILDING_AND_ENVIRONMENT_STYLE = JournalStyle()
