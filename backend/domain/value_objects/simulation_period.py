from pydantic import BaseModel, Field, ConfigDict, model_validator


class SimulationPeriod(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
    )

    start_year: int = Field(
        ..., description="The starting year of the simulation period."
    )
    end_year: int = Field(..., description="The ending year of the simulation period.")
    start_month: int = Field(
        ..., ge=1, le=12, description="The starting month of the simulation period."
    )
    end_month: int = Field(
        ..., ge=1, le=12, description="The ending month of the simulation period."
    )
    start_day: int = Field(
        ..., ge=1, le=31, description="The starting day of the simulation period."
    )
    end_day: int = Field(
        ..., ge=1, le=31, description="The ending day of the simulation period."
    )

    @model_validator(mode="after")
    def validate_period(self):
        if self.start_year > self.end_year:
            raise ValueError("start_year must be less than or equal to end_year")
        if self.start_year == self.end_year:
            if self.start_month > self.end_month:
                raise ValueError(
                    "start_month must be less than or equal to end_month when start_year equals end_year"
                )
            if self.start_month == self.end_month:
                if self.start_day > self.end_day:
                    raise ValueError(
                        "start_day must be less than or equal to end_day when start_year and start_month equal end_year and end_month"
                    )
        return self

    def get_duration_years(self) -> int:
        return self.end_year - self.start_year + 1

    def is_full_year(self) -> bool:
        return (
            self.start_month == 1
            and self.start_day == 1
            and self.end_month == 12
            and self.end_day == 31
        )

    def __str__(self) -> str:
        if self.is_full_year() and self.start_year == self.end_year:
            return f"Year {self.start_year}"

        start = f"{self.start_year}-{self.start_month:02d}-{self.start_day:02d}"
        end = f"{self.end_year}-{self.end_month:02d}-{self.end_day:02d}"
        return f"{start} to {end}"
