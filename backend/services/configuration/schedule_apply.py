from eppy.modeleditor import IDF
from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class ScheduleApply(IApply):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self._config = config

    def apply(self, job: SimulationJob) -> None:
        logger.info("Applying schedule configuration")
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        self._configure_schedule(job.idf)
        logger.info("Schedule configuration applied successfully")

    def _configure_schedule(self, idf: IDF) -> None:
        if idf.getobject("Schedule:Compact", "Always_on") is None:
            idf.newidfobject(
                "Schedule:Compact",
                Name="Always_on",
                Schedule_Type_Limits_Name="On/Off",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 24:00",
                Field_4="1",
            )
            logger.success("Always_on schedule configured successfully")

        if idf.getobject("Schedule:Compact", "Always_off") is None:
            idf.newidfobject(
                "Schedule:Compact",
                Name="Always_off",
                Schedule_Type_Limits_Name="On/Off",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 24:00",
                Field_4="0",
            )
            logger.success("Always_off schedule configured successfully")
