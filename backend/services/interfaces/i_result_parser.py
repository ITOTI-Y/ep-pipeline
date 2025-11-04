from abc import ABC, abstractmethod
from typing import ClassVar

from backend.domain.models import SimulationContext, SimulationResult


class IResultParser(ABC):
    ENERGY_QUERY = """
            SELECT
                RowName,
                ColumnName,
                Value
            FROM
                TabularDataWithStrings
            WHERE
                ReportName = 'AnnualBuildingUtilityPerformanceSummary'
                AND TableName = 'Site and Source Energy'
                AND RowName IN ('Total Site Energy', 'Total Source Energy', 'Net Site Energy', 'Net Source Energy')
                AND ColumnName IN ('Total Energy', 'Energy Per Conditioned Building Area')
                AND Units IN ('kWh', 'kWh/m2')
            """
    AREA_QUERY = """
            SELECT
                RowName,
                ColumnName,
                Value
            FROM
                TabularDataWithStrings
            WHERE
                ReportName = 'AnnualBuildingUtilityPerformanceSummary'
                AND TableName = 'Building Area'
                AND RowName IN ('Total Building Area', 'Net Conditioned Building Area')
                AND Units = 'm2'
            """
    ENERGY_KEY_MAPPING: ClassVar[dict[str, dict[str, str]]] = {
        "Total Site Energy": {
            "Total Energy": "total_site_energy",
            "Energy Per Conditioned Building Area": "total_site_eui",
        },
        "Total Source Energy": {
            "Total Energy": "total_source_energy",
            "Energy Per Conditioned Building Area": "total_source_eui",
        },
        "Net Site Energy": {
            "Total Energy": "net_site_energy",
            "Energy Per Conditioned Building Area": "net_site_eui",
        },
        "Net Source Energy": {
            "Total Energy": "net_source_energy",
            "Energy Per Conditioned Building Area": "net_source_eui",
        },
    }
    AREA_KEY_MAPPING: ClassVar[dict[str, str]] = {
        "Total Building Area": "total_building_area",
        "Net Conditioned Building Area": "net_building_area",
    }
    @abstractmethod
    def parse(
        self,
        result: SimulationResult,
        context: SimulationContext,
    ) -> SimulationResult:
        pass
