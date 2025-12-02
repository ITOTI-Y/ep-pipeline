from loguru import logger

from backend.models import SimulationJob
from backend.services.interfaces import IFileCleaner
from backend.utils.config import ConfigManager


class FileCleaner(IFileCleaner):
    def __init__(self):
        pass

    def clean(
        self,
        job: SimulationJob,
        config: ConfigManager,
        exclude_files: tuple[str, ...] = (),
    ) -> None:
        cleanup_files = config.simulation.cleanup_files
        cleanup_files = [file for file in cleanup_files if file not in exclude_files]
        deleted_count = 0
        for pattern in cleanup_files:
            file_paths = job.output_directory.glob(pattern)
            for file_path in file_paths:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Deleted: {file_path}")
                    deleted_count += 1
                else:
                    logger.warning(f"File not found: {file_path}")

        logger.info(f"Cleaned up {deleted_count} files for job {job.output_directory}")
