from idfpy import IDF
from idfpy.models import (
    ElectricLoadCenterStorageBattery,
    ElectricLoadCenterStorageSimple,
)
from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class StorageApply(IApply):
    def __init__(self, config: ConfigManager, building_type: str):
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
        self._remove_objects(idf, ElectricLoadCenterStorageSimple)
        self._remove_objects(idf, ElectricLoadCenterStorageBattery)

        idf.add(
            ElectricLoadCenterStorageSimple(
                name="PV_Storage",
                availability_schedule_name="Always_on"
                if self._config.storage.capacity[self._building_type] > 0
                else "Always_off",
                radiative_fraction_for_zone_heat_gains=0.0,
                nominal_energetic_efficiency_for_charging=0.95,
                nominal_discharging_energetic_efficiency=0.95,
                maximum_storage_capacity=self._config.storage.capacity[
                    self._building_type
                ]
                * 3600000,
                maximum_power_for_discharging=self._config.storage.max_power[
                    self._building_type
                ]
                * 1000,
                maximum_power_for_charging=self._config.storage.max_power[
                    self._building_type
                ]
                * 1000,
                initial_state_of_charge=self._config.storage.capacity[
                    self._building_type
                ]
                * 3600000
                * 0.5,
            )
        )

        logger.success("Storage configured successfully")
