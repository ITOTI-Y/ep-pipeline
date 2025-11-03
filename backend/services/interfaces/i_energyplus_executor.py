from abc import ABC, abstractmethod
from pathlib import Path

from eppy.modeleditor import IDF

from backend.domain.models import ExecutionResult


class IEnergyPlusExecutor(ABC):
    """EnergyPlus executor interface"""

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
