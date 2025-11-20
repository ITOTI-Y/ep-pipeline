from loguru import logger

from backend.models import SimulationJob
from backend.services.interfaces import IFileCleaner
from backend.utils.config import ConfigManager


class FileCleaner(IFileCleaner):
    def __init__(self):
        self._logger = logger.bind(module=self.__class__.__name__)

    def clean(
        self,
        job: SimulationJob,
        config: ConfigManager,
    ) -> None:
        cleanup_files = config.simulation.cleanup_files
        deleted_count = 0
        for pattern in cleanup_files:
            file_paths = job.output_directory.glob(pattern)
            for file_path in file_paths:
                if file_path.exists():
                    file_path.unlink()
                    self._logger.debug(f"Deleted: {file_path}")
                    deleted_count += 1
                else:
                    self._logger.warning(f"File not found: {file_path}")

        self._logger.info(
            f"Cleaned up {deleted_count} files for job {job.output_directory}"
        )
