from pydantic import BaseModel, ConfigDict, Field


class Location(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    city: str = Field(..., min_length=1, description="Name of the city.")
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude of the location in decimal degrees.",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude of the location in decimal degrees.",
    )

    def get_coordinates(self) -> tuple[float, float] | None:
        return (self.latitude, self.longitude)

    def __str__(self) -> str:
        return f"{self.city} ({self.latitude}, {self.longitude})"
