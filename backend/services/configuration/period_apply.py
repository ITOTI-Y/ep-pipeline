from idfpy import IDF
from idfpy.models import RunPeriod
from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class PeriodApply(IApply):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self._config = config

    def apply(self, job: SimulationJob) -> None:
        logger.info("Applying period configuration")
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        self._configure_simulation_period(job.idf)
        logger.info("Period configuration applied successfully")

    def _configure_simulation_period(self, idf: IDF) -> None:
        self._remove_objects(idf, RunPeriod)

        idf.add(
            RunPeriod(
                name="Default Run Period",
                begin_month=self._config.simulation.begin_month,
                begin_day_of_month=self._config.simulation.begin_day,
                begin_year=self._config.simulation.begin_year,
                end_month=self._config.simulation.end_month,
                end_day_of_month=self._config.simulation.end_day,
                end_year=self._config.simulation.end_year,
            )
        )

        logger.success("Simulation period configured successfully")
