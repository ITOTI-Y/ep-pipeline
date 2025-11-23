from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models.enums import BuildingType


class ECMParameters(BaseModel):
    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
    )

    building_type: BuildingType = Field(
        ..., description="Type of the building (e.g., Residential, Commercial, etc.)"
    )

    window_u_value: float | None = Field(
        default=None,
        ge=0.0,
        description="U-value of the windows in W/m²K",
    )
    window_shgc: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Solar Heat Gain Coefficient (SHGC) of the windows",
    )
    visible_transmittance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Visible transmittance of the windows",
    )
    wall_insulation: float | None = Field(
        default=None,
        ge=0.0,
        description="R-value of the wall insulation in m²K/W",
    )
    infiltration_rate: float | None = Field(
        default=None,
        ge=0.0,
        description="Infiltration rate in air changes per hour (ACH)",
    )
    natural_ventilation_area: float | None = Field(
        default=None,
        ge=0.0,
        description="Area available for natural ventilation in m²",
    )
    cooling_cop: float | None = Field(
        default=None,
        ge=1.0,
        description="Coefficient of Performance (COP) of the cooling system",
    )
    heating_cop: float | None = Field(
        default=None,
        ge=1.0,
        description="Coefficient of Performance (COP) of the heating system",
    )
    cooling_air_temperature: float | None = Field(
        default=None,
        description="Cooling air temperature in °C",
    )
    heating_air_temperature: float | None = Field(
        default=None,
        description="Heating air temperature in °C",
    )
    lighting_power_reduction_level: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Lighting power reduction level (discrete level: 1, 2, 3)",
    )

    @property
    def lighting_power_reduction(self) -> float | None:
        if (
            self.lighting_power_reduction_level is None
            or self.building_type not in self._lighting_power_reduction_map
        ):
            return None

        level_map = self._lighting_power_reduction_map[self.building_type]
        return level_map.get(self.lighting_power_reduction_level, None)

    _lighting_power_reduction_map: dict[BuildingType, dict[int, float]] = {
        BuildingType.OFFICE_LARGE: {1: 0.2, 2: 0.47, 3: 0.53},
        BuildingType.OFFICE_MEDIUM: {1: 0.2, 2: 0.47, 3: 0.53},
        BuildingType.APARTMENT_HIGH_RISE: {1: 0.35, 2: 0.45, 3: 0.55},
        BuildingType.SINGLE_FAMILY_RESIDENTIAL: {1: 0.45, 2: 0.5, 3: 0.64},
        BuildingType.MULTI_FAMILY_RESIDENTIAL: {1: 0.35, 2: 0.45, 3: 0.55},
    }

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def merge(self, other: "ECMParameters") -> "ECMParameters":
        if other.building_type != self.building_type:
            raise ValueError(
                "Cannot merge ECMParameters with different building types."
            )
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return ECMParameters(**merged_dict)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.to_dict().items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ECMParameters):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __str__(self) -> str:
        non_none_params = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"ECMParameters({', '.join(non_none_params)})"
