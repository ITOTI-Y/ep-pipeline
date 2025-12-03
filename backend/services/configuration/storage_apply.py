from eppy.modeleditor import IDF
from loguru import logger

from backend.models import BuildingType, SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class StorageApply(IApply):
    def __init__(self, config: ConfigManager, building_type: BuildingType):
        super().__init__()
        self._config = config
        self._building_type = building_type
    def apply(self, job: SimulationJob) -> None:
        logger.info("Applying storage configuration")
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        self._configure_storage(job.idf)
        logger.info("Storage configuration applied successfully")

    def _configure_storage(self, idf: IDF) -> None:
        self._remove_objects(idf, "ElectricLoadCenter:Storage")

        storage = idf.newidfobject("ElectricLoadCenter:Storage:Simple")
        storage.Name = "PV_Storage"
        storage.Radiative_Fraction_for_Zone_Heat_Gains = 0.0
        storage.Nominal_Energetic_Efficiency_for_Charging = 0.95
        storage.Nominal_Discharging_Energetic_Efficiency = 0.95
        storage.Maximum_Storage_Capacity = self._config.storage.capacity[self._building_type.value] * 3600000 # Convert kWh to J
        storage.Maximum_Power_for_Discharging = 10000
        storage.Maximum_Power_for_Charging = 10000
        storage.Initial_State_of_Charge = 25000

        logger.success("Storage configured successfully")
