from enum import Enum


class BuildingType(str, Enum):
    OFFICE_LARGE = "OfficeLarge"
    OFFICE_MEDIUM = "OfficeMedium"
    OFFICE_SMALL = "OfficeSmall"
    MULTI_FAMILY_RESIDENTIAL = "MultiFamilyResidential"
    SINGLE_FAMILY_RESIDENTIAL = "SingleFamilyResidential"
    HIGH_RISE_RESIDENTIAL = "HighRiseResidential"

    def __str__(self) -> str:
        return self.value


class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        return self.value

    def is_terminal(self) -> bool:
        return self in {
            SimulationStatus.COMPLETED,
            SimulationStatus.FAILED,
            SimulationStatus.CANCELLED,
        }


class SimulationType(str, Enum):
    BASELINE = "baseline"
    PV = "pv"
    OPTIMIZATION = "optimization"
    SENSITIVITY = "sensitivity"
    ECM = "ecm"


class ClimateZone(str, Enum):
    ZONE_1A = "1A"
    ZONE_2A = "2A"
    ZONE_3A = "3A"
    ZONE_4A = "4A"
    ZONE_5A = "5A"
    ZONE_6A = "6A"
    ZONE_7A = "7"
    ZONE_8A = "8"
