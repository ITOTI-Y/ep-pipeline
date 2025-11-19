from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logger.bind(module=__name__)

class Surface(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
    )
    name: str = Field(..., description="The name of the surface.")
    hour_count: int = Field(..., description="The number of hours the surface simulation was run.")
    sum_irradiation: float = Field(..., description="The sum of the irradiation on the surface in kWh/m².")
    unit: str = Field(..., description="The unit of the irradiation.")
    type: str = Field(..., description="The type of the surface.")

class SimulationResult(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
    )

    job_id: UUID = Field(
        ..., description="The unique identifier of the associated simulation job."
    )
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the simulation result.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the simulation result was created.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages encountered during the simulation.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of warning messages encountered during the simulation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata related to the simulation result.",
    )

    table_csv_path: Path | None = Field(
        default=None,
        description="Path to the CSV file containing tabular simulation results.",
    )
    meter_csv_path: Path | None = Field(
        default=None,
        description="Path to the CSV file containing meter simulation results.",
    )
    variables_csv_path: Path | None = Field(
        default=None,
        description="Path to the CSV file containing variables simulation results.",
    )
    sql_path: Path | None = Field(
        default=None,
        description="Path to the SQL file containing detailed simulation data.",
    )
    total_building_area: float | None = Field(
        default=None,
        ge=0,
        description="Total building area (m²) - calculated by ResultParser",
    )
    net_building_area: float | None = Field(
        default=None,
        ge=0,
        description="Net building area (m²) - calculated by ResultParser",
    )
    total_source_energy: float | None = Field(
        default=None,
        ge=0,
        description="Total source energy consumption (kWh) - calculated by ResultParser",
    )
    net_source_energy: float | None = Field(
        default=None,
        ge=0,
        description="Net source energy consumption (kWh) - calculated by ResultParser",
    )
    total_site_energy: float | None = Field(
        default=None,
        ge=0,
        description="Total site energy consumption (kWh) - calculated by ResultParser",
    )
    net_site_energy: float | None = Field(
        default=None,
        ge=0,
        description="Net site energy consumption (kWh) - calculated by ResultParser",
    )
    total_source_eui: float | None = Field(
        default=None,
        ge=0,
        description="Source energy intensity (kWh/m²/yr) - calculated by ResultParser",
    )
    net_source_eui: float | None = Field(
        default=None,
        ge=0,
        description="Net source energy intensity (kWh/m²/yr) - calculated by ResultParser",
    )
    total_site_eui: float | None = Field(
        default=None,
        ge=0,
        description="Total site energy intensity (kWh/m²/yr) - calculated by ResultParser",
    )
    net_site_eui: float | None = Field(
        default=None,
        ge=0,
        description="Net site energy intensity (kWh/m²/yr) - calculated by ResultParser",
    )
    surfaces: list[Surface] = Field(
        default_factory=list,
        description="The list of surfaces in the simulation.",
    )
    success: bool = Field(
        default=False,
        description="Indicates whether the simulation completed successfully.",
    )

    @field_validator(
        "table_csv_path", "meter_csv_path", "variables_csv_path", "sql_path"
    )
    def validate_file_paths(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            logger.warning(f"File path does not exist: {v}")
            return None
        return v

    def add_error(self, message: str) -> None:
        """Add an error message to the simulation result."""
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """Add a warning message to the simulation result."""
        self.warnings.append(message)

    def has_errors(self) -> bool:
        """Check if there are any error messages."""
        return len(self.errors) > 0


    def get_eui_summary(self) -> dict[str, float | None]:
        """Get a summary of energy use intensities."""
        return {
            "total_source_eui": self.total_source_eui,
            "total_site_eui": self.total_site_eui,
            "net_source_eui": self.net_source_eui,
            "net_site_eui": self.net_site_eui,
        }

    def __str__(self) -> str:
        status = "Success" if self.success else "Failed"
        return (
            f"SimulationResult(id={self.id}, job_id={self.job_id}, status={status}, "
            f"total_source_eui={self.total_source_eui}, total_site_eui={self.total_site_eui})"
        )
