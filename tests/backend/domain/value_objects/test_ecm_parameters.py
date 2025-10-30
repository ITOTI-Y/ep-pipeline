"""
Unit tests for ECMParameters value object.

Tests cover:
- Parameter validation and constraints
- Property calculations
- Immutability
- Hash and equality
- Merge operations
- Dictionary conversion
- String representation
"""
import pytest
from pydantic import ValidationError

from backend.domain.value_objects.ecm_parameters import ECMParameters
from backend.domain.models.enums import BuildingType


class TestECMParametersCreation:
    """Test ECMParameters creation and validation."""

    def test_create_with_building_type_only(self):
        """Test creating ECMParameters with only required building_type."""
        params = ECMParameters(building_type=BuildingType.OFFICE_LARGE)
        
        assert params.building_type == BuildingType.OFFICE_LARGE
        assert params.window_u_value is None
        assert params.window_shgc is None
        assert params.visible_transmittance is None
        assert params.wall_insulation is None
        assert params.infiltration_rate is None
        assert params.natural_ventilation_area is None
        assert params.cop is None
        assert params.cooling_air_temperature is None
        assert params.lighting_power_reduction_level is None

    def test_create_with_all_valid_parameters(self):
        """Test creating ECMParameters with all valid parameters."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_MEDIUM,
            window_u_value=1.5,
            window_shgc=0.4,
            visible_transmittance=0.7,
            wall_insulation=2.5,
            infiltration_rate=0.5,
            natural_ventilation_area=10.0,
            cop=4.0,
            cooling_air_temperature=13.0,
            lighting_power_reduction_level=2,
        )
        
        assert params.building_type == BuildingType.OFFICE_MEDIUM
        assert params.window_u_value == 1.5
        assert params.window_shgc == 0.4
        assert params.visible_transmittance == 0.7
        assert params.wall_insulation == 2.5
        assert params.infiltration_rate == 0.5
        assert params.natural_ventilation_area == 10.0
        assert params.cop == 4.0
        assert params.cooling_air_temperature == 13.0
        assert params.lighting_power_reduction_level == 2


class TestECMParametersValidation:
    """Test validation constraints on ECMParameters."""

    def test_window_u_value_negative_fails(self):
        """Test that negative window U-value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                window_u_value=-1.0,
            )
        assert "window_u_value" in str(exc_info.value)

    def test_window_shgc_below_zero_fails(self):
        """Test that SHGC below 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                window_shgc=-0.1,
            )
        assert "window_shgc" in str(exc_info.value)

    def test_window_shgc_above_one_fails(self):
        """Test that SHGC above 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                window_shgc=1.5,
            )
        assert "window_shgc" in str(exc_info.value)

    def test_window_shgc_boundary_values(self):
        """Test SHGC boundary values (0 and 1) are valid."""
        params_zero = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_shgc=0.0,
        )
        assert params_zero.window_shgc == 0.0
        
        params_one = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_shgc=1.0,
        )
        assert params_one.window_shgc == 1.0

    def test_visible_transmittance_below_zero_fails(self):
        """Test that visible transmittance below 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                visible_transmittance=-0.1,
            )
        assert "visible_transmittance" in str(exc_info.value)

    def test_visible_transmittance_above_one_fails(self):
        """Test that visible transmittance above 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                visible_transmittance=1.1,
            )
        assert "visible_transmittance" in str(exc_info.value)

    def test_wall_insulation_negative_fails(self):
        """Test that negative wall insulation raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                wall_insulation=-1.0,
            )
        assert "wall_insulation" in str(exc_info.value)

    def test_infiltration_rate_negative_fails(self):
        """Test that negative infiltration rate raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                infiltration_rate=-0.5,
            )
        assert "infiltration_rate" in str(exc_info.value)

    def test_natural_ventilation_area_negative_fails(self):
        """Test that negative natural ventilation area raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                natural_ventilation_area=-5.0,
            )
        assert "natural_ventilation_area" in str(exc_info.value)

    def test_cop_below_one_fails(self):
        """Test that COP below 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                cop=0.5,
            )
        assert "cop" in str(exc_info.value)

    def test_cop_exactly_one_is_valid(self):
        """Test that COP of exactly 1.0 is valid."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cop=1.0,
        )
        assert params.cop == 1.0

    def test_lighting_power_reduction_level_below_one_fails(self):
        """Test that lighting level below 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                lighting_power_reduction_level=0,
            )
        assert "lighting_power_reduction_level" in str(exc_info.value)

    def test_lighting_power_reduction_level_above_three_fails(self):
        """Test that lighting level above 3 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                lighting_power_reduction_level=4,
            )
        assert "lighting_power_reduction_level" in str(exc_info.value)

    def test_lighting_power_reduction_level_boundary_values(self):
        """Test lighting level boundary values (1, 2, 3) are valid."""
        for level in [1, 2, 3]:
            params = ECMParameters(
                building_type=BuildingType.OFFICE_LARGE,
                lighting_power_reduction_level=level,
            )
            assert params.lighting_power_reduction_level == level


class TestLightingPowerReductionProperty:
    """Test the lighting_power_reduction calculated property."""

    def test_lighting_power_reduction_none_when_level_none(self):
        """Test that property returns None when level is None."""
        params = ECMParameters(building_type=BuildingType.OFFICE_LARGE)
        assert params.lighting_power_reduction is None

    def test_lighting_power_reduction_office_large_level_1(self):
        """Test lighting reduction for OFFICE_LARGE at level 1."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=1,
        )
        assert params.lighting_power_reduction == 0.2

    def test_lighting_power_reduction_office_large_level_2(self):
        """Test lighting reduction for OFFICE_LARGE at level 2."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=2,
        )
        assert params.lighting_power_reduction == 0.47

    def test_lighting_power_reduction_office_large_level_3(self):
        """Test lighting reduction for OFFICE_LARGE at level 3."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=3,
        )
        assert params.lighting_power_reduction == 0.53

    def test_lighting_power_reduction_office_medium_all_levels(self):
        """Test lighting reduction for OFFICE_MEDIUM at all levels."""
        for level, expected in [(1, 0.2), (2, 0.47), (3, 0.53)]:
            params = ECMParameters(
                building_type=BuildingType.OFFICE_MEDIUM,
                lighting_power_reduction_level=level,
            )
            assert params.lighting_power_reduction == expected

    def test_lighting_power_reduction_apartment_high_rise_all_levels(self):
        """Test lighting reduction for APARTMENT_HIGH_RISE at all levels."""
        for level, expected in [(1, 0.35), (2, 0.45), (3, 0.55)]:
            params = ECMParameters(
                building_type=BuildingType.APARTMENT_HIGH_RISE,
                lighting_power_reduction_level=level,
            )
            assert params.lighting_power_reduction == expected

    def test_lighting_power_reduction_single_family_all_levels(self):
        """Test lighting reduction for SINGLE_FAMILY_RESIDENTIAL at all levels."""
        for level, expected in [(1, 0.45), (2, 0.5), (3, 0.64)]:
            params = ECMParameters(
                building_type=BuildingType.SINGLE_FAMILY_RESIDENTIAL,
                lighting_power_reduction_level=level,
            )
            assert params.lighting_power_reduction == expected

    def test_lighting_power_reduction_multi_family_all_levels(self):
        """Test lighting reduction for MULTI_FAMILY_RESIDENTIAL at all levels."""
        for level, expected in [(1, 0.35), (2, 0.45), (3, 0.55)]:
            params = ECMParameters(
                building_type=BuildingType.MULTI_FAMILY_RESIDENTIAL,
                lighting_power_reduction_level=level,
            )
            assert params.lighting_power_reduction == expected


class TestECMParametersImmutability:
    """Test that ECMParameters is immutable (frozen)."""

    def test_cannot_modify_building_type(self):
        """Test that building_type cannot be modified after creation."""
        params = ECMParameters(building_type=BuildingType.OFFICE_LARGE)
        with pytest.raises(ValidationError):
            params.building_type = BuildingType.OFFICE_MEDIUM

    def test_cannot_modify_window_u_value(self):
        """Test that window_u_value cannot be modified after creation."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        with pytest.raises(ValidationError):
            params.window_u_value = 2.0

    def test_cannot_add_new_attributes(self):
        """Test that new attributes cannot be added."""
        params = ECMParameters(building_type=BuildingType.OFFICE_LARGE)
        with pytest.raises(ValidationError):
            params.new_attribute = "value" # type: ignore


class TestECMParametersToDictMethod:
    """Test the to_dict method."""

    def test_to_dict_excludes_none_values(self):
        """Test that to_dict excludes None values."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        result = params.to_dict()
        
        assert "building_type" in result
        assert "window_u_value" in result
        assert "window_shgc" not in result
        assert "wall_insulation" not in result

    def test_to_dict_includes_all_set_values(self):
        """Test that to_dict includes all set values."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            cop=4.0,
        )
        result = params.to_dict()
        
        assert result == {
            "building_type": "OfficeLarge",
            "window_u_value": 1.5,
            "window_shgc": 0.4,
            "cop": 4.0,
        }

    def test_to_dict_with_lighting_level(self):
        """Test to_dict includes lighting_power_reduction_level but not the calculated property."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            lighting_power_reduction_level=2,
        )
        result = params.to_dict()
        
        assert result["lighting_power_reduction_level"] == 2
        # The calculated property is not included in to_dict
        assert "lighting_power_reduction" not in result


class TestECMParametersMergeMethod:
    """Test the merge method."""

    def test_merge_with_same_building_type(self):
        """Test merging parameters with same building type."""
        params1 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        params2 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_shgc=0.4,
        )
        
        merged = params1.merge(params2)
        
        assert merged.building_type == BuildingType.OFFICE_LARGE
        assert merged.window_u_value == 1.5
        assert merged.window_shgc == 0.4

    def test_merge_overwrites_existing_values(self):
        """Test that merge overwrites existing values from first parameter."""
        params1 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            cop=3.0,
        )
        params2 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            cop=4.0,
        )
        
        merged = params1.merge(params2)
        
        assert merged.cop == 4.0
        assert merged.window_u_value == 1.5

    def test_merge_different_building_types_raises_error(self):
        """Test that merging different building types raises ValueError."""
        params1 = ECMParameters(building_type=BuildingType.OFFICE_LARGE)
        params2 = ECMParameters(building_type=BuildingType.OFFICE_MEDIUM)
        
        with pytest.raises(ValueError, match="Cannot merge ECMParameters with different building types"):
            params1.merge(params2)

    def test_merge_preserves_immutability(self):
        """Test that merge creates a new object and doesn't modify originals."""
        params1 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        params2 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_shgc=0.4,
        )
        
        merged = params1.merge(params2)
        
        # Original parameters should be unchanged
        assert params1.window_shgc is None
        assert params2.window_u_value is None
        
        # Merged should have both
        assert merged.window_u_value == 1.5
        assert merged.window_shgc == 0.4


class TestECMParametersHashAndEquality:
    """Test hash and equality methods."""

    def test_equal_parameters_have_same_hash(self):
        """Test that equal parameters have the same hash."""
        params1 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        params2 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        
        assert params1 == params2
        assert hash(params1) == hash(params2)

    def test_different_parameters_not_equal(self):
        """Test that different parameters are not equal."""
        params1 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        params2 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=2.0,
        )
        
        assert params1 != params2

    def test_parameters_can_be_used_in_set(self):
        """Test that parameters can be used in a set (hashable)."""
        params1 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        params2 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        params3 = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=2.0,
        )
        
        param_set = {params1, params2, params3}
        assert len(param_set) == 2  # params1 and params2 are equal

    def test_equality_with_non_ecmparameters(self):
        """Test equality comparison with non-ECMParameters object."""
        params = ECMParameters(building_type=BuildingType.OFFICE_LARGE)
        
        assert params != "not an ECMParameters"
        assert params != 42
        assert params is not None


class TestECMParametersStringRepresentation:
    """Test the __str__ method."""

    def test_str_with_single_parameter(self):
        """Test string representation with single parameter."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        result = str(params)
        
        assert "ECMParameters(" in result
        assert "building_type=OfficeLarge" in result
        assert "window_u_value=1.5" in result

    def test_str_with_multiple_parameters(self):
        """Test string representation with multiple parameters."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
            window_shgc=0.4,
            cop=4.0,
        )
        result = str(params)
        
        assert "ECMParameters(" in result
        assert "building_type=OfficeLarge" in result
        assert "window_u_value=1.5" in result
        assert "window_shgc=0.4" in result
        assert "cop=4.0" in result

    def test_str_excludes_none_values(self):
        """Test that string representation excludes None values."""
        params = ECMParameters(
            building_type=BuildingType.OFFICE_LARGE,
            window_u_value=1.5,
        )
        result = str(params)
        
        assert "window_shgc" not in result
        assert "wall_insulation" not in result