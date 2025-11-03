from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID

from backend.domain.models import SimulationResult


class IResultParser(ABC):
    @abstractmethod
    def parse(
        self,
        job_id: UUID,
        output_directory: Path,
        output_prefix: str,
    ) -> SimulationResult:
        pass
