from eppy.modeleditor import IDF

from backend.models import SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class PeriodApply(IApply):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self._config = config

    def apply(self, job: SimulationJob) -> None:
        self._logger.info("Applying period configuration")
        self._configure_simulation_period(job.idf)
        self._logger.info("Period configuration applied successfully")

    def _configure_simulation_period(self, idf: IDF) -> None:
        self._remove_objects(idf, "RUNPERIOD")

        idf.newidfobject(
            "RUNPERIOD",
            Name="Default Run Period",
            Begin_Month=self._config.simulation.begin_month,
            Begin_Day_of_Month=self._config.simulation.begin_day,
            Begin_Year=self._config.simulation.begin_year,
            End_Month=self._config.simulation.end_month,
            End_Day_of_Month=self._config.simulation.end_day,
            End_Year=self._config.simulation.end_year,
        )

        self._logger.success("Simulation period configured successfully")
