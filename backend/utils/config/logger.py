import sys
from pathlib import Path

from loguru import logger


def set_logger(log_dir: Path) -> None:
    logger.remove()

    logger.add(
        log_dir / "info.log",
        level="INFO",
        rotation="100 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        diagnose=True,
    )

    logger.add(
        log_dir / "error.log",
        level="ERROR",
        rotation="100 MB",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
        diagnose=True,
    )

    logger.add(
        sys.stdout,  # type: ignore[arg-type]
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        enqueue=True,
        diagnose=True,
    )
