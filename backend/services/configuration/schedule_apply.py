from idfpy import IDF
from idfpy.models import ScheduleCompact, ScheduleCompactDataItem
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
        if idf.all_of_type(ScheduleCompact).get("Always_on") is None:
            idf.add(
                ScheduleCompact(
                    name="Always_on",
                    schedule_type_limits_name="On/Off",
                    data=[
                        ScheduleCompactDataItem(
                            field="Through: 12/31",
                        ),
                        ScheduleCompactDataItem(
                            field="For: AllDays",
                        ),
                        ScheduleCompactDataItem(
                            field="Until: 24:00",
                        ),
                        ScheduleCompactDataItem(
                            field="1",
                        ),
                    ],
                )
            )
            logger.success("Always_on schedule configured successfully")

        if idf.all_of_type(ScheduleCompact).get("Always_off") is None:
            idf.add(
                ScheduleCompact(
                    name="Always_off",
                    schedule_type_limits_name="On/Off",
                    data=[
                        ScheduleCompactDataItem(
                            field="Through: 12/31",
                        ),
                        ScheduleCompactDataItem(
                            field="For: AllDays",
                        ),
                        ScheduleCompactDataItem(
                            field="Until: 24:00",
                        ),
                        ScheduleCompactDataItem(
                            field="0",
                        ),
                    ],
                )
            )
            logger.success("Always_off schedule configured successfully")
