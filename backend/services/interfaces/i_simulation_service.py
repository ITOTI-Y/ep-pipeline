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
        pass

    @abstractmethod
    def cleanup(self, context: Tcontext) -> None:
        pass

    @abstractmethod
    def run(self, context: Tcontext) -> SimulationResult:
        try:
            self.prepare(context)
            result = self.execute(context)
            return result
        finally:
            self.cleanup(context)