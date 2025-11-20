from abc import ABC, abstractmethod

from backend.models import SimulationJob, SimulationResult


class IEnergyPlusExecutor(ABC):
    """EnergyPlus executor interface"""

    @abstractmethod
    def run(
        self,
        job: SimulationJob,
    ) -> SimulationResult:
        """
        running energyplus simulation

        Args:
            job (SimulationJob): simulation job

        Returns:
            SimulationResult: simulation result
        """
        pass
