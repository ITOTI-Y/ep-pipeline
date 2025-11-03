from abc import ABC, abstractmethod
from pathlib import Path


class IFileCleaner(ABC):
    @abstractmethod
    def clean(
        self,
        directory: Path,
        cleanup_patterns: list[str],
    ):
        pass
