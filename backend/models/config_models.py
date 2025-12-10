from datetime import date
from pathlib import Path

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
    ecm_dir: Path = Field(..., description="ECM simulation output directory")
    optimization_dir: Path = Field(
        ..., description="Optimization result output directory"
    )
    log_dir: Path = Field(..., description="Log directory")
    eplus_executable: Path = Field(..., description="EnergyPlus executable path")
    idd_file: Path = Field(..., description="IDD file path")
    temp_dir: Path = Field(..., description="Temporary directory")
    idf_files: list[Path] = Field(default_factory=list, description="IDF files")
    tmy_files: list[Path] = Field(default_factory=list, description="TMY weather files")
    ftmy_files: list[Path] = Field(
        default_factory=list, description="Future TMY weather files"
    )

    @field_validator("eplus_executable", "idd_file")
    def validate_file_exists(cls, v: Path) -> Path:
        """Validate file exists"""
        if not v.exists():
            raise ValueError(f"File does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v

    @field_validator(
        "prototype_dir",
        "tmy_dir",
        "ftmy_dir",
        "output_dir",
        "baseline_dir",
        "pv_dir",
        "ecm_dir",
        "optimization_dir",
        "temp_dir",
        "log_dir",
    )
    def validate_directory_exists(cls, v: Path) -> Path:
        """Create directory if not exists and Validate directory exists"""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        elif not v.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return v

    @model_validator(mode="after")
    def initialize_idf_and_weather_files(self) -> "PathsConfig":
        """Initialize idf and weather files"""
        if not self.idf_files:
            object.__setattr__(
                self, "idf_files", list(self.prototype_dir.glob("*.idf"))
            )
        if not self.tmy_files:
            object.__setattr__(self, "tmy_files", list(self.tmy_dir.glob("*.epw")))
        if not self.ftmy_files:
            object.__setattr__(self, "ftmy_files", list(self.ftmy_dir.glob("*.epw")))
        return self


class SimulationConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )

    begin_year: int = Field(..., ge=1900, le=2100, description="Simulation start year")
    end_year: int = Field(..., ge=1900, le=2100, description="Simulation end year")
    begin_month: int = Field(..., ge=1, le=12, description="Simulation start month")
    end_month: int = Field(..., ge=1, le=12, description="Simulation end month")
    begin_day: int = Field(..., ge=1, le=31, description="Simulation start day")
    end_day: int = Field(..., ge=1, le=31, description="Simulation end day")
    default_output_suffix: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Default output file suffix",
    )
    cleanup_files: list[str] = Field(
        ...,
        description="Files to cleanup after simulation",
    )

    @model_validator(mode="after")
    def validate_period(self):
        if self.begin_year > self.end_year:
            raise ValueError("start_year must be less than or equal to end_year")
        if self.begin_year == self.end_year:
            if self.begin_month > self.end_month:
                raise ValueError(
                    "start_month must be less than or equal to end_month when start_year equals end_year"
                )
            if self.begin_month == self.end_month and self.begin_day > self.end_day:
                raise ValueError(
                    "start_day must be less than or equal to end_day when start_year and start_month equal end_year and end_month"
                )
        try:
            _ = date(self.begin_year, self.begin_month, self.begin_day)
            _ = date(self.end_year, self.end_month, self.end_day)
        except ValueError as e:
            raise ValueError(f"Invalid date in simulation period: {e}") from e
        return self

    def get_duration_years(self) -> int:
        return self.end_year - self.begin_year + 1

    def is_full_year(self) -> bool:
        return (
            self.begin_month == 1
            and self.begin_day == 1
            and self.end_month == 12
            and self.end_day == 31
        )

    def __str__(self) -> str:
        if self.is_full_year() and self.begin_year == self.end_year:
            return f"Year {self.begin_year}"

        start = f"{self.begin_year}-{self.begin_month:02d}-{self.begin_day:02d}"
        end = f"{self.end_year}-{self.end_month:02d}-{self.end_day:02d}"
        return f"{start} to {end}"


class ECMParametersConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )
    window_shgc: list[float] = Field(
        default_factory=list, description="Window SHGC values"
    )
    window_u_value: list[float] = Field(
        default_factory=list, description="Window U-value values"
    )
    visible_transmittance: list[float] = Field(
        default_factory=list, description="Visible transmittance values"
    )
    wall_insulation: list[float] = Field(
        default_factory=list, description="Wall insulation values"
    )
    infiltration_rate: list[float] = Field(
        default_factory=list, description="Infiltration rate values"
    )
    natural_ventilation_area: list[float] = Field(
        default_factory=list, description="Natural ventilation area values"
    )
    cooling_cop: list[float] = Field(
        default_factory=list, description="Cooling COP values"
    )
    heating_cop: list[float] = Field(
        default_factory=list, description="Heating COP values"
    )
    cooling_air_temperature: list[float] = Field(
        default_factory=list, description="Cooling air temperature values"
    )
    heating_air_temperature: list[float] = Field(
        default_factory=list, description="Heating air temperature values"
    )
    lighting_power_reduction_level: list[int] = Field(
        default_factory=list, description="Lighting power reduction level values"
    )

    @property
    def keys(self) -> list[str]:
        return list(self.model_dump().keys())


class PVConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )
    radiation_threshold: float = Field(default=800.0, description="Radiation threshold")


class StorageConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )
    capacity: dict = Field(default_factory=lambda: {}, description="Storage capacity")


class GeneticAlgorithmConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
    )
    population_size: int = Field(default=100, description="Population size")
    generations: int = Field(default=100, description="Generations")
    crossover_prob_start: float = Field(
        default=0.9, description="Crossover probability at start"
    )
    crossover_prob_end: float = Field(
        default=0.6, description="Crossover probability at end"
    )
    mutation_prob_start: float = Field(
        default=0.3, description="Mutation probability at start"
    )
    mutation_prob_end: float = Field(
        default=0.05, description="Mutation probability at end"
    )
    gene_crossover_prob: float = Field(
        default=0.5, description="Per-gene crossover probability"
    )
    gene_mutation_prob: float = Field(
        default=0.1, description="Per-gene mutation probability"
    )
    hall_of_fame_percentage: float = Field(
        default=0.1, description="Hall of fame percentage"
    )


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
    )

    seed: int = Field(default=0, description="Random seed")
    genetic: GeneticAlgorithmConfig = Field(
        default_factory=GeneticAlgorithmConfig, description="Genetic configuration"
    )
