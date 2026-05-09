from abc import ABC, abstractmethod

from idfpy import IDF
from idfpy.idf import IDFBaseModel
from loguru import logger

from backend.models import SimulationJob


class IApply(ABC):
    @abstractmethod
    def apply(self, job: SimulationJob) -> None:
        pass

    def _remove_objects(self, idf: IDF, object_type: type[IDFBaseModel]) -> None:
        objects = idf.all_of_type(object_type)
        for name, obj in objects.items():
            idf.remove(object_type, name)
            logger.debug(f"Removed {object_type} object: {obj}")
