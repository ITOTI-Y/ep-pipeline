"""
Figure generation modules for PV simulation visualization.
"""

from .base import BaseFigure
from .storage import StorageSOCFigure, TypicalDayStorageFigure, MonthlyCurtailmentFigure
from .pv import MonthlyPVGenerationFigure, ClimateComparisonFigure, LoadPVMatchFigure
from .ecm import ECMSensitivityHeatmap, SensitivityIndexFigure
from .energy import EUIWaterfallFigure, ClimateEUITrendFigure, SelfConsumptionFigure, PeakReductionFigure
from .supplementary import (
    EnergyBreakdownFigure,
    ECMCorrelationMatrix,
    ClimateResilienceRadar,
    PVEfficiencyTempFigure,
    EnergySankeyFigure,
    LoadClusteringFigure,
    CarbonReductionFigure,
    PerformanceRadarFigure,
)

__all__ = [
    "BaseFigure",
    "StorageSOCFigure",
    "TypicalDayStorageFigure",
    "MonthlyCurtailmentFigure",
    "MonthlyPVGenerationFigure",
    "ClimateComparisonFigure",
    "LoadPVMatchFigure",
    "ECMSensitivityHeatmap",
    "SensitivityIndexFigure",
    "EUIWaterfallFigure",
    "ClimateEUITrendFigure",
    "SelfConsumptionFigure",
    "PeakReductionFigure",
    "EnergyBreakdownFigure",
    "ECMCorrelationMatrix",
    "ClimateResilienceRadar",
    "PVEfficiencyTempFigure",
    "EnergySankeyFigure",
    "LoadClusteringFigure",
    "CarbonReductionFigure",
    "PerformanceRadarFigure",
]
