from eppy.modeleditor import IDF
from abc import ABC, abstractmethod

from ..value_objects.ecm_parameters import ECMParameters

class IECMApplicator(ABC):

    @abstractmethod
    def apply(self, idf: IDF, parameters: ECMParameters) -> None:
        """
        apply ecm parameters to idf file

        Args:
            idf (IDF): IDF object
            parameters (ECMParameters): ECM parameters to apply

        Raises:
            ValueError: if parameters are invalid
            RuntimeError: if application fails
        """

    @abstractmethod
    def validate(self, parameters: ECMParameters) -> bool:
        """
        Verify parameters availability

        Args:
            parameters (ECMParameters): ECM parameters to validate

        Returns:
            bool: True if parameters are valid, False otherwise
        """