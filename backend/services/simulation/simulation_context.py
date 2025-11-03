from pathlib import Path
from typing import Any, Dict

from eppy.modeleditor import IDF
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.domain.models import SimulationJob


class SimulationContext(BaseModel):
    """
    Simulation Context base class

    Args:
        job (SimulationJob): The simulation job object.
        idf (IDF): The IDF object.
        working_directory (Path): The working directory.
        metadata (Dict[str, Any], optional): Additional metadata. Defaults to None.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        frozen=False,
    )

    job: SimulationJob = Field(..., description="The simulation job object.")
    idf: IDF = Field(..., description="The IDF object.")
    working_directory: Path = Field(..., description="The working directory.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata."
    )

    @field_validator("working_directory")
    def validate_working_directory(cls, v: Path) -> Path:
        """
        Validate the working directory

        Args:
            v (Path): The working directory.

        Returns:
            Path: The validated working directory.
        """
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v


class BaselineContext(SimulationContext):
    pass


class PVContext(SimulationContext):
    pass


class ECMContext(SimulationContext):
    pass
