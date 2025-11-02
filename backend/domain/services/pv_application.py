import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd
from eppy.modeleditor import IDF
from loguru import logger

from ..models.simulation_result import SimulationResult


class IPVApplicator(ABC):
    @abstractmethod
    def _find_suitable_surfaces(
        self,
        idf: IDF,
        baseline_result: SimulationResult,
        min_irradiance_threshold: float = 800.0,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def apply(
        self,
        idf: IDF,
        baseline_result: SimulationResult,
        panel_efficiency: float,
        inverter_efficiency: float,
        min_irradiance_threshold: float = 800.0,
    ) -> None:
        pass


class PVApplicator(IPVApplicator):
    def __init__(self) -> None:
        self._logger = logger.bind(service=self.__class__.__name__)

    def apply(
        self,
        idf: IDF,
        baseline_result: SimulationResult,
        panel_efficiency: float,
        inverter_efficiency: float,
        min_irradiance_threshold: float = 800.0,
    ) -> None:
        if not baseline_result.success:
            self._logger.error("Baseline simulation failed")
            raise RuntimeError("Baseline simulation failed")

        suitable_surfaces = self._find_suitable_surfaces(
            idf,
            baseline_result,
            min_irradiance_threshold,
        )

        if not suitable_surfaces:
            self._logger.error("No suitable surfaces found for PV application")
            raise RuntimeError("No suitable surfaces found for PV application")

        appiled_surfaces = []
        total_area = 0.0
        total_capacity = 0.0

        for surface_info in suitable_surfaces:
            area = surface_info["area"]
            title = surface_info["title"]
            azimuth = surface_info["azimuth"]
            surface_name = surface_info["surface_name"]
        pass
