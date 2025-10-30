"""
Shared pytest fixtures for the test suite.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from backend.domain.models.enums import BuildingType


@pytest.fixture
def mock_idf():
    """Create a mock IDF object with common methods."""
    idf = MagicMock()
    idf.idfobjects = MagicMock()
    idf.newidfobject = MagicMock()
    idf.getobject = MagicMock(return_value=None)
    idf.removeallidfobjects = MagicMock()
    return idf


@pytest.fixture
def sample_building_type():
    """Provide a sample BuildingType for testing."""
    return BuildingType.OFFICE_LARGE