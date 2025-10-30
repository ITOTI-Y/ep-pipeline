from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class SimulationResult(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
    )

    job_id: UUID = Field(
        ..., description="The unique identifier of the associated simulation job."
    )
    output_directory: Path = Field(
        ..., description="The directory where the simulation results are stored."
    )
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the simulation result.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the simulation result was created.",
    )
    error_messages: List[str] = Field(
        default_factory=list,
        description="List of error messages encountered during the simulation.",
    )
    warning_messages: List[str] = Field(
        default_factory=list,
        description="List of warning messages encountered during the simulation.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata related to the simulation result.",
    )

    table_csv_path: Optional[Path] = Field(
        default=None,
        description="Path to the CSV file containing tabular simulation results.",
    )
    meter_csv_path: Optional[Path] = Field(
        default=None,
        description="Path to the CSV file containing meter simulation results.",
    )
    sql_path: Optional[Path] = Field(
        default=None,
        description="Path to the SQL file containing detailed simulation data.",
    )
    source_eui: Optional[float] = Field(
        default=None,
        ge=0,
        description="Source energy intensity (kWh/m²/yr) - calculated by ResultParser",
    )
    site_eui: Optional[float] = Field(
        default=None,
        ge=0,
        description="Site energy intensity (kWh/m²/yr) - calculated by ResultParser",
    )
    total_energy_kwh: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total energy consumption (kWh) - calculated by ResultParser",
    )
    execution_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total execution time of the simulation (seconds).",
    )
    success: bool = Field(
        default=False,
        description="Indicates whether the simulation completed successfully.",
    )

    def add_error(self, message: str) -> None:
        """Add an error message to the simulation result."""
        self.error_messages.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """Add a warning message to the simulation result."""
        self.warning_messages.append(message)

    def has_errors(self) -> bool:
        """Check if there are any error messages."""
        return len(self.error_messages) > 0

    def is_valid(self) -> bool:
        """Check if the simulation result is valid (no errors)."""
        if not self.success:
            return False

        if self.has_errors():
            return False

        if self.source_eui is None:
            return False

        return True

    def get_eui_summary(self) -> Dict[str, Optional[float]]:
        """Get a summary of energy use intensities."""
        return {
            "source_eui": self.source_eui,
            "site_eui": self.site_eui,
            "total_energy_kwh": self.total_energy_kwh,
        }

    def __str__(self) -> str:
        status = "Success" if self.success else "Failed"
        return (
            f"SimulationResult(id={self.id}, job_id={self.job_id}, status={status}, "
            f"source_eui={self.source_eui})"
        )
