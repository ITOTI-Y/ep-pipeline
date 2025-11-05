from abc import ABC, abstractmethod

from backend.domain.models import SimulationContext
from backend.utils.config import ConfigManager


class IFileCleaner(ABC):
    @abstractmethod
    def clean(
        self,
        context: SimulationContext,
        config: ConfigManager,
    ):
        pass
