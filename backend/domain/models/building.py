from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict

from .enums import BuildingType


class Building(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    name: str = Field(..., description="The name of the building.")
    building_type: BuildingType = Field(
        ..., description="The type of the building."
    )
    location: str = Field(..., description="The location of the building.")
    idf_file_path: Path = Field(
        ..., description="The file path to the building's IDF file."
    )
    id: UUID = Field(
        default_factory=uuid4, description="The unique identifier for the building."
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="The timestamp when the building was created.",
    )
    modified_at: datetime = Field(
        default_factory=datetime.now,
        description="The timestamp when the building was last modified.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the building."
    )
    num_floors: Optional[int] = Field(
        None, description="The number of floors in the building."
    )

    @field_validator("idf_file_path")
    def validate_idf_file_path(cls, v: Path) -> Path:
        if not v.exists() or not v.is_file():
            raise ValueError(
                f"The IDF file path '{v}' does not exist or is not a file."
            )
        if v.suffix.lower() != ".idf":
            raise ValueError(f"The file '{v}' is not an IDF file.")
        return v

    def get_identifier(self) -> str:
        """Returns a string identifier for the building."""
        return f"{self.location}_{self.building_type.value}"
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Updates the metadata dictionary with a new key-value pair."""
        self.metadata[key] = value
        self.modified_at = datetime.now()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieves a value from the metadata dictionary by key."""
        return self.metadata.get(key, default)
    
    def __str__(self) -> str:
        return (
            f"Building(name='{self.name}',"
            f"type={self.building_type.value},"
            f"location='{self.location}')"
        )