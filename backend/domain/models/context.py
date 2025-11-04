from typing import Any

from eppy.modeleditor import IDF
from pydantic import BaseModel, ConfigDict, Field

from .simulation_job import SimulationJob


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
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata."
    )


class BaselineContext(SimulationContext):
    pass


class PVContext(SimulationContext):
    pass


class ECMContext(SimulationContext):
    pass
