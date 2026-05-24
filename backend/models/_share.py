from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

BUILDING_TYPES: Final[dict[str, str]] = {
    "coa": "Commercial Office A",
    "cob": "Commercial Office B",
    "goa": "Government Office A",
    "gob": "Government Office B",
    "highs": "High Rise Apartment slab type",
    "hight": "High Rise Apartment tower type",
    "inp": "Inpatient",
    "lh": "Large Hotel",
    "sh": "Small Hotel",
    "low": "Low Rise Apartment",
    "mall": "Shopping Mall",
    "outp": "Outpatient",
    "sch": "Primary/secondary school",
    "th": "Terraced House",
    "uni": "University",
}


class IDFFile(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    city: str = Field(..., description="City name")
    building_type: str = Field(..., description="Building type")
    year: int = Field(..., description="Year")
    file_path: Path = Field(..., description="Path to the IDF file")

    @field_validator("building_type")
    def parse_building_type(cls, v: str) -> str:
        if v.lower() not in BUILDING_TYPES:
            raise ValueError(f"Invalid building type: {v}")
        return v.lower()

    @field_validator("city")
    def normalize_city_name(cls, v: str) -> str:
        return v.lower()

    def __str__(self) -> str:
        return f"{self.city}_{self.building_type}_{self.year}"


class WeatherFile(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    file_path: Path = Field(..., description="Path to the weather file")
    province: str = Field(..., description="Province name")
    city: str = Field(..., description="City name")
    wmo_id: int = Field(..., description="WMO ID")
    code: str = Field(..., description="Weather code")

    @field_validator("city", "province")
    def normalize_city_name(cls, v: str) -> str:
        return v.lower()

    @field_validator("code")
    def validate_weather_code(cls, v: str) -> str:
        if v.lower() not in [
            "tmy",
            "ssp126",
            "ssp245",
            "ssp370",
            "ssp434",
            "ssp585",
        ]:
            raise ValueError(f"Invalid weather code: {v}")
        return v.lower()

    def __str__(self) -> str:
        return f"{self.province}_{self.city}_{self.wmo_id}"
