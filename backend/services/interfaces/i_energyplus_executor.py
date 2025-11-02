from abc import ABC, abstractmethod
from pathlib import Path

from eppy.modeleditor import IDF


class ExecutionResult:
    """EnergyPlus execution result"""

    def __init__(
        self,
        success: bool,
        return_code: int,
        stdout: str,
        stderr: str,
        output_directory: Path,
    ):
        self.success = success
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.output_directory = output_directory
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, message: str) -> None:
        """Add an error message to the execution result."""
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """Add a warning message to the execution result."""
        self.warnings.append(message)


class IEnergyPlusExecutor(ABC):
    """EnergyPlus executor interface"""

    @abstractmethod
    @abstractmethod
    def run(
        self,
        idf: IDF,
        weather_file: Path,
        output_directory: Path,
        output_prefix: str,
        read_variables: bool = True,
    ) -> ExecutionResult:
        """
        running energyplus simulation

        Args:
            idf (IDF): IDF object
            weather_file (Path): weather file path
            output_directory (Path): output directory
            output_prefix (str): output file prefix
            read_variables (bool, optional): whether to read output variables. Defaults to True.
        """
        pass

    @abstractmethod
    def validate_installation(self) -> bool:
        """Validate EnergyPlus installation"""
        pass
