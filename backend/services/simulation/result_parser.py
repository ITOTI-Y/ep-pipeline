from pathlib import Path
from uuid import UUID

from backend.domain.models import SimulationResult
from backend.services.interfaces import IResultParser


class ResultParser(IResultParser):
    def parse(
        self,
        job_id: UUID,
        output_directory: Path,
        output_prefix: str,
    ) -> SimulationResult:
        return SimulationResult(
            job_id=job_id,
            output_directory=output_directory,
        )
