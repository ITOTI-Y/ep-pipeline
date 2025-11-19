from abc import ABC, abstractmethod

from backend.models import SimulationContext, SimulationResult


class IEnergyPlusExecutor(ABC):
    """EnergyPlus executor interface"""

    @abstractmethod
    def run(
        self,
        context: SimulationContext,
    ) -> SimulationResult:
        """
        running energyplus simulation

        Args:
            context (SimulationContext): simulation context

        Returns:
            SimulationResult: simulation result
        """
        pass
