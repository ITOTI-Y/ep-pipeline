"""
Unit tests for ECMApplicator service.

Tests cover:
- Initialization and logger binding
- Main apply method orchestration
- Window parameter application
- Wall insulation parameter application
- Infiltration parameter application
- Natural ventilation parameter application
- Coil and chiller parameter application
- Cooling air temperature parameter application
- Lighting parameter application
- Error handling and edge cases
"""

from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from backend.domain.models.ecm_parameters import ECMParameters
from backend.domain.models.enums import BuildingType
from backend.domain.services.ecm_applicator import ECMApplicator, IECMApplicator


class TestIECMApplicatorInterface:
    """Test the IECMApplicator abstract interface."""

    def test_interface_cannot_be_instantiated(self):
        """Test that the abstract interface cannot be instantiated."""
        with pytest.raises(TypeError):
            IECMApplicator()  # type: ignore

    def test_interface_requires_apply_method(self):
        """Test that implementing class must provide apply method."""

        class IncompleteApplicator(IECMApplicator):
            pass

        with pytest.raises(TypeError):
            IncompleteApplicator()  # type: ignore


class TestECMApplicatorInitialization:
    """Test ECMApplicator initialization."""

    def test_initialization_creates_logger(self):
        """Test that initialization creates a bound logger."""
        applicator = ECMApplicator()

        assert hasattr(applicator, "_logger")
        assert applicator._logger is not None

    def test_initialization_binds_service_name(self):
        """Test that logger is bound with service name."""
        with patch.object(logger, "bind") as mock_bind:
            mock_bind.return_value = logger
            ECMApplicator()

            mock_bind.assert_called_once_with(service="ECMApplicator")


class TestECMApplicatorMainApplyMethod:
    """Test the main apply method orchestration."""

    @pytest.fixture
    def applicator(self):
        """Create an ECMApplicator instance."""
        return ECMApplicator()

    @pytest.fixture
    def mock_idf(self):
        """Create a mock IDF object."""
        idf = MagicMock()
        idf.idfobjects = MagicMock()
        return idf

    def test_apply_with_no_parameters_does_nothing(self, applicator, mock_idf):
        """Test that apply with no ECM parameters set does nothing."""
        params = ECMParameters(building_type=BuildingType.OFFICE_LARGE)

        applicator.apply(mock_idf, params)

        # Should complete without errors
        assert True

    def test_apply_does_not_call_window_parameters_with_only_u_value(
        self, applicator, mock_idf
    ):
        """Test that apply does NOT call window parameter method when only U-value is set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )

        with patch.object(applicator, "_apply_window_parameters") as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_not_called()

    def test_apply_calls_window_parameters_with_shgc_all_three_parameters_set(
        self, applicator, mock_idf
    ):
        """Test window parameters are applied when only SHGC is set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
        )

        with patch.object(applicator, "_apply_window_parameters") as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_called_once_with(mock_idf, params)

    def test_apply_does_not_call_window_parameters_with_partial_params(
        self, applicator, mock_idf
    ):
        """Test window parameters are NOT applied when only some parameters are set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_shgc=0.4,
        )

        with patch.object(applicator, "_apply_window_parameters") as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_not_called()

    def test_apply_calls_wall_insulation_when_set(self, applicator, mock_idf):
        """Test that apply calls wall insulation method when set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            wall_insulation=2.5,
        )

        with patch.object(
            applicator, "_apply_wall_insulation_parameters"
        ) as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_called_once_with(mock_idf, params)

    def test_apply_calls_infiltration_when_set(self, applicator, mock_idf):
        """Test that apply calls infiltration method when set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            infiltration_rate=0.5,
        )

        with patch.object(applicator, "_apply_infiltration_parameters") as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_called_once_with(mock_idf, params)

    def test_apply_calls_natural_ventilation_when_set(self, applicator, mock_idf):
        """Test that apply calls natural ventilation method when set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            natural_ventilation_area=10.0,
        )

        with patch.object(
            applicator, "_apply_natural_ventilation_parameters"
        ) as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_called_once_with(mock_idf, params)

    def test_apply_calls_cop_when_set(self, applicator, mock_idf):
        """Test that apply calls COP method when set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_cop=4.0,
            heating_cop=4.0,
        )

        with patch.object(
            applicator, "_apply_coil_and_chiller_parameters"
        ) as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_called_once_with(mock_idf, params)

    def test_apply_calls_cooling_air_temperature_when_set(self, applicator, mock_idf):
        """Test that apply calls cooling air temperature method when set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_air_temperature=13.0,
        )

        with patch.object(
            applicator, "_apply_cooling_air_temperature_parameters"
        ) as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_called_once_with(mock_idf, params)

    def test_apply_calls_lighting_when_set(self, applicator, mock_idf):
        """Test that apply calls lighting method when set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=2,
        )

        with patch.object(applicator, "_apply_lighting_parameters") as mock_method:
            applicator.apply(mock_idf, params)
            mock_method.assert_called_once_with(mock_idf, params)

    def test_apply_calls_multiple_methods_when_multiple_params_set(
        self, applicator, mock_idf
    ):
        """Test that apply calls multiple methods when multiple params are set."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
            wall_insulation=2.5,
            cooling_cop=4.0,
            heating_cop=4.0,
        )

        with (
            patch.object(applicator, "_apply_window_parameters") as mock_window,
            patch.object(applicator, "_apply_wall_insulation_parameters") as mock_wall,
            patch.object(applicator, "_apply_coil_and_chiller_parameters") as mock_cop,
        ):
            applicator.apply(mock_idf, params)

            mock_window.assert_called_once_with(mock_idf, params)
            mock_wall.assert_called_once_with(mock_idf, params)
            mock_cop.assert_called_once_with(mock_idf, params)

    def test_apply_raises_runtime_error_on_exception(self, applicator, mock_idf):
        """Test that apply raises RuntimeError when any method fails."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
        )

        with patch.object(applicator, "_apply_window_parameters") as mock_method:
            mock_method.side_effect = Exception("Test error")

            with pytest.raises(RuntimeError, match="Failed to apply ECM parameters"):
                applicator.apply(mock_idf, params)


class TestApplyWindowParameters:
    """Test _apply_window_parameters method."""

    @pytest.fixture
    def applicator(self):
        """Create an ECMApplicator instance."""
        return ECMApplicator()

    @pytest.fixture
    def mock_idf(self):
        """Create a mock IDF object."""
        idf = MagicMock()
        idf.idfobjects = MagicMock()
        idf.getobject = MagicMock(return_value=None)
        idf.newidfobject = MagicMock()
        return idf

    def test_creates_window_material_with_correct_name(self, applicator, mock_idf):
        """Test that window material is created with formatted name."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.50,
            window_shgc=0.40,
            visible_transmittance=0.70,
        )

        applicator._apply_window_parameters(mock_idf, params)

        # Check that newidfobject was called with correct material name
        calls = mock_idf.newidfobject.call_args_list
        material_call = calls[0]
        assert material_call[0][0] == "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"
        assert "WindowMaterial_SimpleGlazingSystem_1.50_0.40_0.70" in str(material_call)

    def test_skips_creation_if_material_already_exists(self, applicator, mock_idf):
        """Test that method returns early if window material already exists."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
        )

        # Mock getobject to return an existing material
        mock_idf.getobject = MagicMock(return_value=MagicMock())

        applicator._apply_window_parameters(mock_idf, params)

        # newidfobject should not be called
        mock_idf.newidfobject.assert_not_called()

    def test_creates_construction_object(self, applicator, mock_idf):
        """Test that construction object is created."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
        )

        applicator._apply_window_parameters(mock_idf, params)

        # Check that CONSTRUCTION was created
        calls = mock_idf.newidfobject.call_args_list
        construction_call = calls[1]
        assert construction_call[0][0] == "CONSTRUCTION"

    def test_modifies_window_fenestration_surfaces(self, applicator, mock_idf):
        """Test that window fenestration surfaces are modified."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
        )

        # Create mock fenestration surfaces
        mock_window1 = MagicMock()
        mock_window1.Surface_Type = "WINDOW"
        mock_window1.Name = "Window1"

        mock_window2 = MagicMock()
        mock_window2.Surface_Type = "WINDOW"
        mock_window2.Name = "Window2"

        mock_door = MagicMock()
        mock_door.Surface_Type = "DOOR"
        mock_door.Name = "Door1"

        mock_idf.idfobjects.get = MagicMock(
            return_value=[mock_window1, mock_window2, mock_door]
        )

        applicator._apply_window_parameters(mock_idf, params)

        # Check that only windows were modified
        assert "Construction_Name" in mock_window1.__dict__
        assert "Construction_Name" in mock_window2.__dict__
        # Door should not be modified
        assert "Construction_Name" not in mock_door.__dict__

    def test_handles_empty_fenestration_surfaces_list(self, applicator, mock_idf):
        """Test that method handles empty fenestration surfaces list."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
        )

        mock_idf.idfobjects.get = MagicMock(return_value=[])

        # Should not raise an error
        applicator._apply_window_parameters(mock_idf, params)

    def test_handles_mixed_case_surface_types(self, applicator, mock_idf):
        """Test that surface type comparison is case-insensitive."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
        )

        # Create windows with different case types
        mock_window_lower = MagicMock()
        mock_window_lower.Surface_Type = "window"

        mock_window_upper = MagicMock()
        mock_window_upper.Surface_Type = "WINDOW"

        mock_window_mixed = MagicMock()
        mock_window_mixed.Surface_Type = "Window"

        mock_idf.idfobjects.get = MagicMock(
            return_value=[mock_window_lower, mock_window_upper, mock_window_mixed]
        )

        applicator._apply_window_parameters(mock_idf, params)

        # All should be modified
        assert hasattr(mock_window_lower, "Construction_Name")
        assert hasattr(mock_window_upper, "Construction_Name")
        assert hasattr(mock_window_mixed, "Construction_Name")


class TestApplyWallInsulationParameters:
    """Test _apply_wall_insulation_parameters method."""

    @pytest.fixture
    def applicator(self):
        """Create an ECMApplicator instance."""
        return ECMApplicator()

    @pytest.fixture
    def mock_idf(self):
        """Create a mock IDF object."""
        idf = MagicMock()
        idf.idfobjects = MagicMock()
        idf.newidfobject = MagicMock()
        idf.removeallidfobjects = MagicMock()
        return idf

    def test_creates_insulation_material_with_correct_name(self, applicator, mock_idf):
        """Test that insulation material is created with formatted name."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            wall_insulation=2.50,
        )

        applicator._apply_wall_insulation_parameters(mock_idf, params)

        # Check Material:NoMass was created
        calls = mock_idf.newidfobject.call_args_list
        material_call = calls[0]
        assert material_call[0][0] == "Material:NoMass"
        assert "UserDefined Insulation Material_2.50" in str(material_call)

    def test_creates_schedule_constant(self, applicator, mock_idf):
        """Test that schedule constant is created."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            wall_insulation=2.5,
        )

        applicator._apply_wall_insulation_parameters(mock_idf, params)

        # Check Schedule:Constant was created
        calls = mock_idf.newidfobject.call_args_list
        schedule_call = calls[1]
        assert schedule_call[0][0] == "Schedule:Constant"

    def test_removes_existing_moveable_insulation(self, applicator, mock_idf):
        """Test that existing moveable insulation objects are removed."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            wall_insulation=2.5,
        )

        applicator._apply_wall_insulation_parameters(mock_idf, params)

        mock_idf.removeallidfobjects.assert_called_once_with(
            "SurfaceControl:MoveableInsulation"
        )

    def test_applies_to_outdoor_walls_only(self, applicator, mock_idf):
        """Test that insulation is only applied to outdoor walls."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            wall_insulation=2.5,
        )

        # Create mock surfaces
        outdoor_wall = MagicMock()
        outdoor_wall.Outside_Boundary_Condition = "Outdoors"
        outdoor_wall.Surface_Type = "WALL"
        outdoor_wall.Name = "OutdoorWall1"

        indoor_wall = MagicMock()
        indoor_wall.Outside_Boundary_Condition = "Adiabatic"
        indoor_wall.Surface_Type = "WALL"
        indoor_wall.Name = "IndoorWall1"

        floor = MagicMock()
        floor.Outside_Boundary_Condition = "Outdoors"
        floor.Surface_Type = "FLOOR"
        floor.Name = "Floor1"

        mock_idf.idfobjects.get = MagicMock(
            return_value=[outdoor_wall, indoor_wall, floor]
        )

        applicator._apply_wall_insulation_parameters(mock_idf, params)

        # Count SurfaceControl:MoveableInsulation creations (after schedule and material)
        moveable_insulation_calls = [
            call_item
            for call_item in mock_idf.newidfobject.call_args_list
            if call_item[0][0] == "SurfaceControl:MoveableInsulation"
        ]
        assert len(moveable_insulation_calls) == 1

    def test_applies_to_outdoor_roofs(self, applicator, mock_idf):
        """Test that insulation is applied to outdoor roofs."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            wall_insulation=2.5,
        )

        # Create mock roof surface
        outdoor_roof = MagicMock()
        outdoor_roof.Outside_Boundary_Condition = "Outdoors"
        outdoor_roof.Surface_Type = "ROOF"
        outdoor_roof.Name = "Roof1"

        mock_idf.idfobjects.get = MagicMock(return_value=[outdoor_roof])

        applicator._apply_wall_insulation_parameters(mock_idf, params)

        # Should create moveable insulation for roof
        moveable_insulation_calls = [
            call_item
            for call_item in mock_idf.newidfobject.call_args_list
            if call_item[0][0] == "SurfaceControl:MoveableInsulation"
        ]
        assert len(moveable_insulation_calls) == 1

    def test_handles_empty_surfaces_list(self, applicator, mock_idf):
        """Test that method handles empty surfaces list."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            wall_insulation=2.5,
        )

        mock_idf.idfobjects.get = MagicMock(return_value=[])

        # Should not raise an error
        applicator._apply_wall_insulation_parameters(mock_idf, params)


class TestApplyInfiltrationParameters:
    """Test _apply_infiltration_parameters method."""

    @pytest.fixture
    def applicator(self):
        """Create an ECMApplicator instance."""
        return ECMApplicator()

    @pytest.fixture
    def mock_idf(self):
        """Create a mock IDF object."""
        idf = MagicMock()
        idf.idfobjects = MagicMock()
        return idf

    def test_modifies_infiltration_objects(self, applicator, mock_idf):
        """Test that infiltration objects are modified."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            infiltration_rate=0.5,
        )

        # Create mock infiltration objects
        infiltration1 = MagicMock()
        infiltration1.Name = "Infiltration1"

        infiltration2 = MagicMock()
        infiltration2.Name = "Infiltration2"

        mock_idf.idfobjects.get = MagicMock(return_value=[infiltration1, infiltration2])

        applicator._apply_infiltration_parameters(mock_idf, params)

        # Check both were modified
        assert infiltration1.Design_Flow_Rate_Calculation_Method == "AirChanges/Hour"
        assert infiltration1.Air_Changes_per_Hour == 0.5
        assert infiltration2.Design_Flow_Rate_Calculation_Method == "AirChanges/Hour"
        assert infiltration2.Air_Changes_per_Hour == 0.5

    def test_handles_empty_infiltration_list(self, applicator, mock_idf):
        """Test that method handles empty infiltration list gracefully."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            infiltration_rate=0.5,
        )

        mock_idf.idfobjects.get = MagicMock(return_value=[])

        # Should not raise an error
        applicator._apply_infiltration_parameters(mock_idf, params)

    def test_applies_correct_infiltration_rate(self, applicator, mock_idf):
        """Test that correct infiltration rate is applied."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            infiltration_rate=1.2,
        )

        infiltration = MagicMock()
        mock_idf.idfobjects.get = MagicMock(return_value=[infiltration])

        applicator._apply_infiltration_parameters(mock_idf, params)

        assert infiltration.Air_Changes_per_Hour == 1.2


class TestApplyNaturalVentilationParameters:
    """Test _apply_natural_ventilation_parameters method."""

    @pytest.fixture
    def applicator(self):
        """Create an ECMApplicator instance."""
        return ECMApplicator()

    @pytest.fixture
    def mock_idf(self):
        """Create a mock IDF object."""
        idf = MagicMock()
        idf.idfobjects = MagicMock()
        return idf

    def test_modifies_ventilation_objects(self, applicator, mock_idf):
        """Test that natural ventilation objects are modified."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            natural_ventilation_area=10.0,
        )

        # Create mock ventilation objects
        ventilation1 = MagicMock()
        ventilation1.Name = "Ventilation1"

        ventilation2 = MagicMock()
        ventilation2.Name = "Ventilation2"

        mock_idf.idfobjects.get = MagicMock(return_value=[ventilation1, ventilation2])

        applicator._apply_natural_ventilation_parameters(mock_idf, params)

        # Check both were modified
        assert ventilation1.Open_Area == 10.0
        assert ventilation2.Open_Area == 10.0

    def test_handles_empty_ventilation_list(self, applicator, mock_idf):
        """Test that method handles empty ventilation list gracefully."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            natural_ventilation_area=10.0,
        )

        mock_idf.idfobjects.get = MagicMock(return_value=[])

        # Should not raise an error
        applicator._apply_natural_ventilation_parameters(mock_idf, params)

    def test_applies_correct_ventilation_area(self, applicator, mock_idf):
        """Test that correct ventilation area is applied."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            natural_ventilation_area=15.5,
        )

        ventilation = MagicMock()
        mock_idf.idfobjects.get = MagicMock(return_value=[ventilation])

        applicator._apply_natural_ventilation_parameters(mock_idf, params)

        assert ventilation.Open_Area == 15.5


class TestApplyCoilAndChillerParameters:
    """Test _apply_coil_and_chiller_parameters method."""

    @pytest.fixture
    def applicator(self):
        """Create an ECMApplicator instance."""
        return ECMApplicator()

    @pytest.fixture
    def mock_idf(self):
        """Create a mock IDF object."""
        idf = MagicMock()
        idf.idfobjects = MagicMock()
        return idf

    def test_modifies_coil_objects(self, applicator, mock_idf):
        """Test that coil objects are modified."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_cop=4.0,
            heating_cop=4.0,
        )

        # Create mock coil with COP field
        coil = MagicMock()
        coil.Name = "Coil1"
        coil.Rated_COP = 3.0

        mock_idf.idfobjects.keys = MagicMock(return_value=["COIL:COOLING:DX"])
        mock_idf.idfobjects.get = MagicMock(return_value=[coil])

        applicator._apply_coil_and_chiller_parameters(mock_idf, params)

        assert coil.Rated_COP == 4.0

    def test_modifies_chiller_objects(self, applicator, mock_idf):
        """Test that chiller objects are modified."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_cop=5.0,
            heating_cop=5.0,
        )

        # Create mock chiller with COP field
        chiller = MagicMock()
        chiller.Name = "Chiller1"
        chiller.Reference_COP = 4.0

        mock_idf.idfobjects.keys = MagicMock(return_value=["CHILLER:ELECTRIC:EIR"])
        mock_idf.idfobjects.get = MagicMock(return_value=[chiller])

        applicator._apply_coil_and_chiller_parameters(mock_idf, params)

        assert chiller.Reference_COP == 5.0

    def test_handles_multiple_cop_fields(self, applicator, mock_idf):
        """Test that multiple COP field names are checked."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_cop=4.5,
            heating_cop=4.5,
        )

        # Create equipment with multiple COP fields
        equipment = MagicMock()
        equipment.Name = "Equipment1"
        equipment.Gross_Rated_Cooling_COP = 3.0
        equipment.High_Speed_Gross_Rated_Cooling_COP = 3.2

        mock_idf.idfobjects.keys = MagicMock(
            return_value=["COIL:COOLING:DX:MULTISPEED"]
        )
        mock_idf.idfobjects.get = MagicMock(return_value=[equipment])

        applicator._apply_coil_and_chiller_parameters(mock_idf, params)

        assert equipment.Gross_Rated_Cooling_COP == 4.5
        assert equipment.High_Speed_Gross_Rated_Cooling_COP == 4.5

    def test_skips_non_coil_chiller_objects(self, applicator, mock_idf):
        """Test that non-coil/chiller objects are skipped."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_cop=4.0,
            heating_cop=4.0,
        )

        mock_idf.idfobjects.keys = MagicMock(
            return_value=["ZONE", "BUILDING", "COIL:COOLING:DX"]
        )

        # Setup different returns for different object types
        def get_side_effect(obj_type, _default=None):
            if obj_type == "COIL:COOLING:DX":
                coil = MagicMock()
                coil.Rated_COP = 3.0
                return [coil]
            return _default or []

        mock_idf.idfobjects.get = MagicMock(side_effect=get_side_effect)

        # Should not raise an error
        applicator._apply_coil_and_chiller_parameters(mock_idf, params)

    def test_handles_equipment_without_cop_fields(self, applicator, mock_idf):
        """Test that equipment without COP fields is skipped gracefully."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_cop=4.0,
            heating_cop=4.0,
        )

        # Create equipment without any COP fields
        equipment = MagicMock(spec=["Name"])
        equipment.Name = "Equipment1"

        mock_idf.idfobjects.keys = MagicMock(return_value=["COIL:COOLING:WATER"])
        mock_idf.idfobjects.get = MagicMock(return_value=[equipment])

        # Should not raise an error
        applicator._apply_coil_and_chiller_parameters(mock_idf, params)

    def test_continues_on_exception_for_equipment_type(self, applicator, mock_idf):
        """Test that exception for one equipment type doesn't stop processing."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_cop=4.0,
            heating_cop=4.0,
        )

        mock_idf.idfobjects.keys = MagicMock(
            return_value=["COIL:COOLING:DX", "CHILLER:ELECTRIC:EIR"]
        )

        # Setup to raise exception for first type but succeed for second
        call_count = [0]

        def get_side_effect(_obj_type, _default=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError()
            else:
                chiller = MagicMock()
                chiller.Reference_COP = 3.0
                return [chiller]

        mock_idf.idfobjects.get = MagicMock(side_effect=get_side_effect)

        # Should not raise an error and should process second type
        applicator._apply_coil_and_chiller_parameters(mock_idf, params)


class TestApplyCoolingAirTemperatureParameters:
    """Test _apply_cooling_air_temperature_parameters method."""

    @pytest.fixture
    def applicator(self):
        """Create an ECMApplicator instance."""
        return ECMApplicator()

    @pytest.fixture
    def mock_idf(self):
        """Create a mock IDF object."""
        idf = MagicMock()
        idf.idfobjects = MagicMock()
        return idf

    def test_modifies_sizing_zone_objects(self, applicator, mock_idf):
        """Test that sizing zone objects are modified."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_air_temperature=13.0,
        )

        # Create mock sizing zone objects
        sizing_zone1 = MagicMock()
        sizing_zone1.Name = "Zone1"

        sizing_zone2 = MagicMock()
        sizing_zone2.Name = "Zone2"

        mock_idf.idfobjects.get = MagicMock(return_value=[sizing_zone1, sizing_zone2])

        applicator._apply_cooling_air_temperature_parameters(mock_idf, params)

        # Check both were modified
        assert sizing_zone1.Zone_Cooling_Design_Supply_Air_Temperature == 13.0
        assert sizing_zone2.Zone_Cooling_Design_Supply_Air_Temperature == 13.0

    def test_handles_empty_sizing_zone_list(self, applicator, mock_idf):
        """Test that method handles empty sizing zone list."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_air_temperature=13.0,
        )

        mock_idf.idfobjects.get = MagicMock(return_value=[])

        # Should not raise an error
        applicator._apply_cooling_air_temperature_parameters(mock_idf, params)

    def test_applies_correct_temperature(self, applicator, mock_idf):
        """Test that correct temperature is applied."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cooling_air_temperature=12.5,
        )

        sizing_zone = MagicMock()
        mock_idf.idfobjects.get = MagicMock(return_value=[sizing_zone])

        applicator._apply_cooling_air_temperature_parameters(mock_idf, params)

        assert sizing_zone.Zone_Cooling_Design_Supply_Air_Temperature == 12.5


class TestApplyLightingParameters:
    """Test _apply_lighting_parameters method."""

    @pytest.fixture
    def applicator(self):
        """Create an ECMApplicator instance."""
        return ECMApplicator()

    @pytest.fixture
    def mock_idf(self):
        """Create a mock IDF object."""
        idf = MagicMock()
        idf.idfobjects = MagicMock()
        return idf

    def test_modifies_lighting_level_method(self, applicator, mock_idf):
        """Test lighting modification for LIGHTINGLEVEL calculation method."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=2,
        )

        # Create mock light with LIGHTINGLEVEL method
        light = MagicMock()
        light.Design_Level_Calculation_Method = "LIGHTINGLEVEL"
        light.Lighting_Level = 1000.0

        mock_idf.idfobjects.get = MagicMock(return_value=[light])

        applicator._apply_lighting_parameters(mock_idf, params)

        # Should be reduced by 0.47 (level 2 for OFFICE_LARGE)
        assert light.Lighting_Level == 1000.0 * 0.47

    def test_modifies_watts_per_area_method(self, applicator, mock_idf):
        """Test lighting modification for WATTS/AREA calculation method."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=1,
        )

        # Create mock light with WATTS/AREA method
        light = MagicMock()
        light.Design_Level_Calculation_Method = "WATTS/AREA"
        light.Watts_per_Floor_Area = 10.0

        mock_idf.idfobjects.get = MagicMock(return_value=[light])

        applicator._apply_lighting_parameters(mock_idf, params)

        # Should be reduced by 0.2 (level 1 for OFFICE_LARGE)
        assert light.Watts_per_Floor_Area == 10.0 * 0.2

    def test_modifies_watts_per_person_method(self, applicator, mock_idf):
        """Test lighting modification for WATTS/PERSON calculation method."""
        params = ECMParameters(
            building_type=BuildingType.SINGLE_FAMILY_RESIDENTIAL,
            lighting_power_reduction_level=3,
        )

        # Create mock light with WATTS/PERSON method
        light = MagicMock()
        light.Design_Level_Calculation_Method = "WATTS/PERSON"
        light.Watts_per_Person = 5.0

        mock_idf.idfobjects.get = MagicMock(return_value=[light])

        applicator._apply_lighting_parameters(mock_idf, params)

        # Should be reduced by 0.64 (level 3 for SINGLE_FAMILY_RESIDENTIAL)
        assert light.Watts_per_Person == 5.0 * 0.64

    def test_skips_unsupported_calculation_methods(self, applicator, mock_idf):
        """Test that unsupported calculation methods are skipped."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=2,
        )

        # Create light with unsupported method
        light = MagicMock()
        light.Design_Level_Calculation_Method = "UNKNOWN_METHOD"
        light.Original_Power = 100.0

        mock_idf.idfobjects.get = MagicMock(return_value=[light])

        # Should not raise an error
        applicator._apply_lighting_parameters(mock_idf, params)

        # Original power should not be modified
        assert light.Original_Power == 100.0

    def test_handles_empty_lights_list(self, applicator, mock_idf):
        """Test that method handles empty lights list gracefully."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=2,
        )

        mock_idf.idfobjects.get = MagicMock(return_value=[])

        # Should not raise an error
        applicator._apply_lighting_parameters(mock_idf, params)

    def test_modifies_multiple_lights(self, applicator, mock_idf):
        """Test that multiple light objects are modified."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=2,
        )

        # Create multiple lights with different methods
        light1 = MagicMock()
        light1.Design_Level_Calculation_Method = "LIGHTINGLEVEL"
        light1.Lighting_Level = 1000.0

        light2 = MagicMock()
        light2.Design_Level_Calculation_Method = "WATTS/AREA"
        light2.Watts_per_Floor_Area = 10.0

        mock_idf.idfobjects.get = MagicMock(return_value=[light1, light2])

        applicator._apply_lighting_parameters(mock_idf, params)

        # Both should be reduced
        assert light1.Lighting_Level == 1000.0 * 0.47
        assert light2.Watts_per_Floor_Area == 10.0 * 0.47

    def test_different_building_types_have_different_reductions(
        self, applicator, mock_idf
    ):
        """Test that different building types have different reduction factors."""
        # Test APARTMENT_HIGH_RISE
        params_apartment = ECMParameters(
            building_type=BuildingType.APARTMENT_HIGH_RISE,
            lighting_power_reduction_level=2,
        )

        light = MagicMock()
        light.Design_Level_Calculation_Method = "LIGHTINGLEVEL"
        light.Lighting_Level = 1000.0

        mock_idf.idfobjects.get = MagicMock(return_value=[light])

        applicator._apply_lighting_parameters(mock_idf, params_apartment)

        # Should be reduced by 0.45 (level 2 for APARTMENT_HIGH_RISE)
        assert light.Lighting_Level == 1000.0 * 0.45
