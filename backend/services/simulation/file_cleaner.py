from loguru import logger

from backend.domain.models import SimulationContext
from backend.services.interfaces import IFileCleaner
from backend.utils.config import ConfigManager


class FileCleaner(IFileCleaner):
    def __init__(self):
        self._logger = logger.bind(module=self.__class__.__name__)

    def clean(
        self,
        context: SimulationContext,
        config: ConfigManager,
    ) -> None:

        cleanup_files = config.simulation.cleanup_files
        for pattern in cleanup_files:
            file_paths = context.job.output_directory.glob(pattern)
            for file_path in file_paths:
                if file_path.exists():
                    file_path.unlink()
                    self._logger.debug(f"Deleted: {file_path}")
                else:
                    self._logger.warning(f"File not found: {file_path}")

        self._logger.info(f"Cleaned up {len(cleanup_files)} files for job {context.job.id}")
