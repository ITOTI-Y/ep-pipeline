from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PathsConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        frozen=False,
    )

    prototype_dir: Path = Field(..., description="Prototype idf file directory path")
    tmy_dir: Path = Field(..., description="TMY weather file directory")
    ftmy_dir: Path = Field(..., description="Future TMY weather file directory")
    output_dir: Path = Field(..., description="Output root directory")
    baseline_dir: Path = Field(..., description="Baseline simulation output directory")
    pv_dir: Path = Field(..., description="PV simulation output directory")
    optimization_dir: Path = Field(
        ..., description="Optimization result output directory"
    )
    eplus_executable: Path = Field(..., description="EnergyPlus executable path")
    idd_file: Path = Field(..., description="IDD file path")
    temp_dir: Path = Field(..., description="Temporary directory")

    @field_validator("eplus_executable", "idd_file")
    def validate_file_exists(cls, v: Path) -> Path:
        """Validate file exists"""
        if not v.exists():
            raise ValueError(f"File does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v

    @field_validator("prototype_dir", "tmy_dir", "ftmy_dir", "output_dir", "baseline_dir", "pv_dir", "optimization_dir", "temp_dir")
    def validate_directory_exists(cls, v: Path) -> Path:
        """Create directory if not exists and Validate directory exists"""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v



class SimulationConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )

    start_year: int = Field(..., ge=1900, le=2100, description="Simulation start year")
    end_year: int = Field(..., ge=1900, le=2100, description="Simulation end year")
    default_output_suffix: str = Field(
        default=...,
        min_length=1,
        max_length=10,
        description="Default output file suffix",
    )
    cleanup_files: list[str] = Field(
        ...,
        description="Files to cleanup after simulation",
    )

    @model_validator(mode="after")
    def validate_years(self) -> "SimulationConfig":
        """Validate start year <= end year"""
        if self.start_year > self.end_year:
            raise ValueError("Start year must be <= end year")
        return self


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )

    sensitivity: Dict[str, Any] = Field(
        default_factory=dict, description="Sensitivity analysis configuration"
    )
    optimization: Dict[str, Any] = Field(
        default_factory=dict, description="Optimization configuration"
    )
    surrogate_models: Dict[str, Any] = Field(
        default_factory=dict, description="Surrogate models configuration"
    )
