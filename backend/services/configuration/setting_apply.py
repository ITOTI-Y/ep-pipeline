from eppy.modeleditor import IDF
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
        sim_control_list = idf.idfobjects.get("SIMULATIONCONTROL", [])

        for sim_control in sim_control_list:
            sim_control.Do_Zone_Sizing_Calculation = "Yes"
            sim_control.Do_System_Sizing_Calculation = "Yes"
            sim_control.Do_Plant_Sizing_Calculation = "Yes"
            sim_control.Run_Simulation_for_Sizing_Periods = "No"
            sim_control.Run_Simulation_for_Weather_File_Run_Periods = "Yes"
            sim_control.Do_HVAC_Sizing_Simulation_for_Sizing_Periods = "Yes"
            sim_control.Maximum_Number_of_HVAC_Sizing_Simulation_Passes = 2

        logger.success("Setting configuration applied successfully")
