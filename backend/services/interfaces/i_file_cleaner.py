from abc import ABC, abstractmethod

from backend.models import SimulationJob
from backend.utils.config import ConfigManager


class IFileCleaner(ABC):
    @abstractmethod
    def clean(
        self,
        job: SimulationJob,
        config: ConfigManager,
        exclude_files: tuple[str, ...] = (),
    ) -> None:
        pass
