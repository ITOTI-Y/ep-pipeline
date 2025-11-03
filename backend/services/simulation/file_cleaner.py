from pathlib import Path

from backend.services.interfaces import IFileCleaner


class FileCleaner(IFileCleaner):
    def clean(
        self,
        directory: Path,
        cleanup_patterns: list[str],
    ) -> None:
        pass
