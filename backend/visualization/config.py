"""
Figure configuration for Building and Environment journal standards.

Provides figure sizing, font settings, color schemes, and DPI configurations
compliant with Elsevier journal requirements.
"""

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class FigureConfig:
    """
    Building and Environment journal figure configuration.

    Based on Elsevier artwork guidelines:
    - Resolution: 300 DPI (halftone), 500 DPI (combination), 1000 DPI (line art)
    - Font: Times New Roman, 7pt normal, 6pt subscript/superscript
    - Single column: 90mm (3.54 in), Double column: 190mm (7.48 in)
    """

    # Resolution settings (DPI)
    dpi_halftone: int = 300
    dpi_combination: int = 500
    dpi_line_art: int = 1000
    dpi_default: int = 300

    # Figure dimensions (inches)
    # Single column: 90mm = 3.54 inches
    # Double column: 190mm = 7.48 inches
    single_column_width: float = 3.54
    double_column_width: float = 7.48
    single_column_height: float = 2.5
    double_column_height: float = 3.0

    # Font settings
    font_family: str = "Times New Roman"
    font_size_normal: int = 7
    font_size_subscript: int = 6
    font_size_title: int = 8
    font_size_legend: int = 6

    # Output formats
    output_formats: list = field(default_factory=lambda: ["pdf", "png"])
    max_file_size_mb: int = 10

    # Line widths
    line_width_thin: float = 0.5
    line_width_normal: float = 1.0
    line_width_thick: float = 1.5

    # Marker sizes
    marker_size_small: int = 3
    marker_size_normal: int = 5
    marker_size_large: int = 7

    def get_rc_params(self) -> dict:
        """Get matplotlib rcParams for consistent styling."""
        return {
            "font.family": "serif",
            "font.serif": [self.font_family, "DejaVu Serif"],
            "font.size": self.font_size_normal,
            "axes.labelsize": self.font_size_normal,
            "axes.titlesize": self.font_size_title,
            "axes.linewidth": self.line_width_thin,
            "xtick.labelsize": self.font_size_normal,
            "ytick.labelsize": self.font_size_normal,
            "xtick.major.width": self.line_width_thin,
            "ytick.major.width": self.line_width_thin,
            "legend.fontsize": self.font_size_legend,
            "legend.frameon": False,
            "figure.dpi": self.dpi_default,
            "savefig.dpi": self.dpi_combination,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # TrueType fonts for PDF
            "ps.fonttype": 42,
        }


class ColorSchemes:
    """
    Color schemes for academic paper visualizations.

    Uses colorblind-friendly palettes where possible.
    """

    # Climate scenario colors (colorblind-friendly)
    CLIMATE_COLORS: ClassVar[dict[str, str]] = {
        "TMY": "#1f77b4",      # Blue - Baseline
        "SSP126": "#2ca02c",   # Green - Low emission
        "SSP245": "#9467bd",   # Purple - Medium-low emission
        "SSP370": "#ff7f0e",   # Orange - Medium-high emission
        "SSP434": "#d62728",   # Red - Medium emission
        "SSP585": "#8c564b",   # Brown - High emission
    }

    # Building type colors
    BUILDING_COLORS: ClassVar[dict[str, str]] = {
        "OfficeLarge": "#1f77b4",
        "OfficeMedium": "#ff7f0e",
        "ApartmentHighRise": "#2ca02c",
        "SingleFamilyResidential": "#d62728",
        "MultiFamilyResidential": "#9467bd",
    }

    # Building type display names
    BUILDING_NAMES: ClassVar[dict[str, str]] = {
        "OfficeLarge": "Large Office",
        "OfficeMedium": "Medium Office",
        "ApartmentHighRise": "High-Rise Apartment",
        "SingleFamilyResidential": "Single Family",
        "MultiFamilyResidential": "Multi-Family",
    }

    # Energy flow colors
    ENERGY_COLORS: ClassVar[dict[str, str]] = {
        "pv_generation": "#f1c40f",     # Gold - PV generation
        "self_consumed": "#27ae60",     # Green - Self-consumed
        "grid_export": "#3498db",       # Blue - Grid export
        "grid_import": "#e74c3c",       # Red - Grid import
        "curtailed": "#95a5a6",         # Gray - Curtailed
        "storage_charge": "#9b59b6",    # Purple - Charging
        "storage_discharge": "#1abc9c", # Cyan - Discharging
        "demand": "#34495e",            # Dark gray - Demand
    }

    # ECM parameter colors (for sensitivity analysis)
    ECM_COLORS: ClassVar[dict[str, str]] = {
        "window_u_value": "#e41a1c",
        "window_shgc": "#377eb8",
        "visible_transmittance": "#4daf4a",
        "wall_insulation": "#984ea3",
        "infiltration_rate": "#ff7f00",
        "natural_ventilation_area": "#ffff33",
        "cooling_cop": "#a65628",
        "heating_cop": "#f781bf",
        "cooling_air_temperature": "#999999",
        "heating_air_temperature": "#66c2a5",
        "lighting_power_reduction_level": "#fc8d62",
    }

    # ECM parameter display names
    ECM_NAMES: ClassVar[dict[str, str]] = {
        "window_u_value": "Window U-value",
        "window_shgc": "Window SHGC",
        "visible_transmittance": "Visible Trans.",
        "wall_insulation": "Wall Insulation",
        "infiltration_rate": "Infiltration Rate",
        "natural_ventilation_area": "Natural Vent. Area",
        "cooling_cop": "Cooling COP",
        "heating_cop": "Heating COP",
        "cooling_air_temperature": "Cooling Air Temp.",
        "heating_air_temperature": "Heating Air Temp.",
        "lighting_power_reduction_level": "Lighting Reduction",
    }

    # Diverging colormap for heatmaps
    DIVERGING_CMAP: ClassVar[str] = "RdBu_r"

    # Sequential colormap for positive values
    SEQUENTIAL_CMAP: ClassVar[str] = "YlOrRd"

    @classmethod
    def get_climate_color(cls, scenario: str) -> str:
        """Get color for a climate scenario."""
        return cls.CLIMATE_COLORS.get(scenario, "#333333")

    @classmethod
    def get_building_color(cls, building: str) -> str:
        """Get color for a building type."""
        return cls.BUILDING_COLORS.get(building, "#333333")

    @classmethod
    def get_building_name(cls, building: str) -> str:
        """Get display name for a building type."""
        return cls.BUILDING_NAMES.get(building, building)

    @classmethod
    def get_ecm_name(cls, param: str) -> str:
        """Get display name for an ECM parameter."""
        return cls.ECM_NAMES.get(param, param)


# Storage capacity configuration (kWh)
STORAGE_CAPACITY: dict[str, float] = {
    "OfficeLarge": 0.0,
    "OfficeMedium": 0.0,
    "ApartmentHighRise": 0.0,
    "SingleFamilyResidential": 13.0,
    "MultiFamilyResidential": 20.0,
}

# Buildings with storage systems
BUILDINGS_WITH_STORAGE: list[str] = [
    "SingleFamilyResidential",
    "MultiFamilyResidential",
]

# All building types
ALL_BUILDINGS: list[str] = [
    "OfficeLarge",
    "OfficeMedium",
    "ApartmentHighRise",
    "SingleFamilyResidential",
    "MultiFamilyResidential",
]

# All climate scenarios
ALL_CLIMATES: list[str] = [
    "TMY",
    "SSP126",
    "SSP245",
    "SSP370",
    "SSP434",
    "SSP585",
]

# ECM parameters list
ECM_PARAMETERS: list[str] = [
    "window_u_value",
    "window_shgc",
    "visible_transmittance",
    "wall_insulation",
    "infiltration_rate",
    "natural_ventilation_area",
    "cooling_cop",
    "heating_cop",
    "cooling_air_temperature",
    "heating_air_temperature",
    "lighting_power_reduction_level",
]

# EUI targets for sensitivity analysis
EUI_TARGETS: list[str] = [
    "total_site_eui",
    "net_site_eui",
    "total_source_eui",
]

# Grid emission factor (kgCO2/kWh) - China grid average
GRID_EMISSION_FACTOR: float = 0.581
