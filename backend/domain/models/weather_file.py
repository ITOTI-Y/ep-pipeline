import re
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WeatherFile(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    file_path: Path = Field(..., description="Path to the weather file")
    location: str = Field(..., description="Location name of the weather data")
    scenario: str | None  = Field(
        default=None,
        description="Scenario associated with the weather data (e.g., TMY, FTMY, SSP etc.)",
    )
    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the weather file"
    )
    is_future: bool | None = Field(
        default=False,
        description="Indicates if the weather file represents future climate data",
    )

    @field_validator("file_path")
    def validate_file_path(cls, v: Path) -> Path:
        if not v.exists() or not v.is_file():
            raise ValueError(f"File path {v} does not exist or is not a file.")
        if v.suffix.lower() != ".epw":
            raise ValueError(f"File {v} is not an EPW weather file.")
        return v

    @model_validator(mode="after")
    def get_scenario_description(self) -> "WeatherFile":
        """Return a human-readable description of the weather file scenario."""
        if not self.scenario:
            raw = self.file_path.stem.strip().upper().replace(" ", "")
            digits = re.sub(r"[^0-9]", "", raw.replace("SSP", ""))
            scenario_map: dict[str, str] = {
                "126": "SSP1-2.6 (Low Emissions)",
                "245": "SSP2-4.5 (Intermediate Emissions)",
                "370": "SSP3-7.0 (Medium High Emissions)",
                "434": "SSP4-3.4 (Intermediate Emissions, low overshoot)",
                "585": "SSP5-8.5 (High Emissions)",
            }
            self.scenario = scenario_map.get(digits, f"Future Scenario {digits}")

        return self

    def get_identifier(self) -> str:
        """Generate a unique identifier for the weather file based on location and scenario."""
        return f"{self.location}_{self.scenario}"

    def __str__(self) -> str:
        return f"WeatherFile(location='{self.location}', scenario='{self.scenario}')"
