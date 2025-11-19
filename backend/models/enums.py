from enum import Enum


class BuildingType(str, Enum):
    OFFICE_LARGE = "OfficeLarge"
    OFFICE_MEDIUM = "OfficeMedium"
    MULTI_FAMILY_RESIDENTIAL = "MultiFamilyResidential"
    SINGLE_FAMILY_RESIDENTIAL = "SingleFamilyResidential"
    APARTMENT_HIGH_RISE = "ApartmentHighRise"

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
