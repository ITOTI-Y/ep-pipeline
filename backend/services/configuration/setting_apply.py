from idfpy import IDF
from idfpy.models import SimulationControl, SizingPlant
from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class SettingApply(IApply):
    def __init__(self, config: ConfigManager):
        self._config = config

    def apply(self, job: SimulationJob) -> None:
        logger.info("Applying setting configuration")
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        self._configure_setting(job.idf)
        logger.info("Setting configuration applied successfully")

    def _configure_setting(self, idf: IDF) -> None:
        sim_control_list = idf.all_of_type(SimulationControl)
        has_sizing_plant = bool(idf.all_of_type(SizingPlant))

        for sim_control in sim_control_list.values():
            sim_control.do_zone_sizing_calculation = "Yes"
            sim_control.do_system_sizing_calculation = "Yes"
            sim_control.do_plant_sizing_calculation = (
                "Yes" if has_sizing_plant else "No"
            )
            sim_control.run_simulation_for_sizing_periods = "No"
            sim_control.run_simulation_for_weather_file_run_periods = "Yes"
            sim_control.do_hvac_sizing_simulation_for_sizing_periods = (
                "Yes" if has_sizing_plant else "No"
            )
            sim_control.maximum_number_of_hvac_sizing_simulation_passes = 2

        logger.success("Setting configuration applied successfully")
