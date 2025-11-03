from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from backend.domain.models import SimulationResult

Tcontext = TypeVar("Tcontext")


class ISimulationService(ABC, Generic[Tcontext]):
    @abstractmethod
    def prepare(self, context: Tcontext) -> None:
        """
        Prepare the simulation context.

        include:
            - create output directory
            - validate files existence
            - setting output variables
            - apply preparation logic

        Args:
            context (Tcontext): The simulation context.

        Raises:
            ValidationError: If validation fails.
            FileNotFoundError: If required files are missing.
            PreparationError: If preparation process fails.
        """
        pass

    @abstractmethod
    def execute(self, context: Tcontext) -> SimulationResult:
        """
        Execute the simulation.

        Args:
            context (Tcontext): The simulation context.

        Returns:
            SimulationResult: The simulation result containing output paths,
                energy metrics, and execution metadata.

        Raises:
            SimulationError: If the simulation execution fails.
            RuntimeError: If EnergyPlus encounters a runtime error.
        """
        pass

    @abstractmethod
    def cleanup(self, context: Tcontext) -> None:
        """
        Clean up temporary files and resources after simulation.

        This method should remove intermediate files and release any
        resources held during the simulation. It should not raise exceptions.

        Args:
            context (Tcontext): The simulation context.
        """
        pass

    def run(self, context: Tcontext) -> SimulationResult:
        try:
            self.prepare(context)
            result = self.execute(context)
            return result
        finally:
            self.cleanup(context)
