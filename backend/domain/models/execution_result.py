from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ExecutionResult(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
    )

    success: bool = Field(
        default=False, description="Whether the execution was successful."
    )
    return_code: int = Field(default=1, description="The return code of the execution.")
    stdout: str = Field(default="", description="The standard output of the execution.")
    stderr: str = Field(default="", description="The standard error of the execution.")
    output_directory: Path = Field(
        ..., description="The directory where the execution results are stored."
    )
    errors: list[str] = Field(
        default_factory=list, description="The errors of the execution."
    )
    warnings: list[str] = Field(
        default_factory=list, description="The warnings of the execution."
    )

    

    def add_error(self, message: str) -> None:
        """Add an error message to the execution result."""
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """Add a warning message to the execution result."""
        self.warnings.append(message)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
