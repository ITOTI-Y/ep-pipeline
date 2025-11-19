from abc import ABC, abstractmethod

from eppy.modeleditor import IDF
from loguru import logger

from backend.models import SimulationContext


class IApply(ABC):
    def __init__(self):
        self._logger = logger.bind(service=self.__class__.__name__)

    @abstractmethod
    def apply(self, context: SimulationContext) -> None:
        pass

    def _remove_objects(self, idf: IDF, object_type: str) -> None:
        objects = list(idf.idfobjects.get(object_type, []))
        for obj in objects:
            idf.removeidfobject(obj)
            self._logger.debug(f"Removed {object_type} object: {obj}")
