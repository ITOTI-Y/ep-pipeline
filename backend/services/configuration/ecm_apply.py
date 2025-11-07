from eppy.modeleditor import IDF

from backend.domain.models import ECMParameters, SimulationContext
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class ECMApply(IApply):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self._config = config

    def apply(self, context: SimulationContext, parameters: ECMParameters) -> None:
        self._logger.info("Applying ECM configuration")
        self._apply_window_parameters(context, parameters)
        self._apply_wall_insulation_parameters(context, parameters)
        self._apply_infiltration_parameters(context, parameters)
        self._apply_natural_ventilation_parameters(context, parameters)
        self._apply_cooling_coil_and_chiller_parameters(context, parameters)
        self._apply_heating_coil_and_chiller_parameters(context, parameters)
        self._apply_cooling_air_temperature_parameters(context, parameters)
        self._apply_heating_air_temperature_parameters(context, parameters)
        self._apply_lighting_parameters(context, parameters)
        self._logger.info("ECM configuration applied successfully")

    def _apply_window_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply windows parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        window_material_name = (
            "WindowMaterial_SimpleGlazingSystem"
            + f"_{parameters.window_u_value:.2f}"
            + f"_{parameters.window_shgc:.2f}"
            + f"_{parameters.visible_transmittance:.2f}"
        )

        if (
            idf.getobject("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", window_material_name)
            is not None
        ):
            return

        idf.newidfobject(
            "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
            Name=window_material_name,
            UFactor=parameters.window_u_value,
            Solar_Heat_Gain_Coefficient=parameters.window_shgc,
            Visible_Transmittance=parameters.visible_transmittance,
        )

        constructions_name = f"Construction_window_{window_material_name}"

        idf.newidfobject(
            "CONSTRUCTION",
            Name=constructions_name,
            Outside_Layer=window_material_name,
        )

        fenestration_surfaces = idf.idfobjects.get("FENESTRATIONSURFACE:DETAILED", [])
        modified_count = 0
        for surface in fenestration_surfaces:
            if surface.Surface_Type.upper() == "WINDOW":
                surface.Construction_Name = constructions_name
                self._logger.debug(
                    f"Set construction name to {constructions_name} for {surface.Name}"
                )
                modified_count += 1

        self._logger.info(f"Modified {modified_count} fenestration surface objects")

    def _apply_wall_insulation_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply wall insulation parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        insulation_materials_name = (
            "UserDefined Insulation Material" + f"_{parameters.wall_insulation:.2f}"
        )

        idf.newidfobject(
            "Material:NoMass",
            Name=insulation_materials_name,
            Roughness="Rough",
            Thermal_Resistance=parameters.wall_insulation,
            Thermal_Absorptance=0.9,
            Solar_Absorptance=0.7,
            Visible_Absorptance=0.7,
        )

        schedule_name = "WallInsulationSchedule_AlwaysOn"
        idf.newidfobject(
            "Schedule:Constant",
            Name=schedule_name,
            Hourly_Value=1.0,
        )

        self._remove_objects(idf, "SurfaceControl:MoveableInsulation")
        surfaces = idf.idfobjects.get("BUILDINGSURFACE:DETAILED", [])
        modified_count = 0
        for surface in surfaces:
            if (
                surface.Outside_Boundary_Condition == "Outdoors"
                and surface.Surface_Type.upper() in ["WALL", "ROOF"]
            ):
                idf.newidfobject(
                    "SurfaceControl:MovableInsulation",
                    Insulation_Type="Outside",
                    Surface_Name=surface.Name,
                    Material_Name=insulation_materials_name,
                    Schedule_Name=schedule_name,
                )
                self._logger.debug(
                    f"Set insulation material to {insulation_materials_name} for {surface.Name}"
                )
                modified_count += 1

        self._logger.info(f"Modified {modified_count} surface control objects")

    def _apply_infiltration_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply infiltration parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        infiltration_objects = idf.idfobjects.get("ZONEINFILTRATION:DESIGNFLOWRATE", [])
        modified_count = 0

        if not infiltration_objects:
            self._logger.warning(
                "No ZONEINFILTRATION:DESIGNFLOWRATE objects found in IDF"
            )
            return

        for infiltration in infiltration_objects:
            infiltration.Design_Flow_Rate_Calculation_Method = "AirChanges/Hour"
            infiltration.Air_Changes_per_Hour = parameters.infiltration_rate
            self._logger.debug(
                f"Set infiltration rate to {parameters.infiltration_rate} ACH for {infiltration.Name}"
            )
            modified_count += 1

        self._logger.info(f"Modified {modified_count} infiltration objects")

    def _apply_natural_ventilation_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply natural ventilation parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        self._remove_objects(idf, "ZONEVENTILATION:WindandStackOpenArea")
        zones = idf.idfobjects.get("ZONE", [])
        modified_count = 0

        for zone in zones:
            idf.newidfobject(
                "ZONEVENTILATION:WindandStackOpenArea",
                Name=f"WindandStackOpenArea_{zone.Name}_{parameters.natural_ventilation_area:.2f}",
                Zone_or_Space_Name=zone.Name,
                Opening_Area=parameters.natural_ventilation_area,
            )
            self._logger.debug(
                f"Set natural ventilation area to {parameters.natural_ventilation_area} m² for {zone.Name}"
            )
            modified_count += 1

        self._logger.info(f"Modified {modified_count} ventilation objects")

    def _apply_cooling_coil_and_chiller_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply cooling coil and chiller parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        modified_count = 0

        cop_field_names = [
            "Gross_Rated_Cooling_COP",
            "Reference_COP",
            "Rated_COP",
            "High_Speed_Gross_Rated_Cooling_COP",
            "Low_Speed_Gross_Rated_Cooling_COP",
            "Rated_COP_at_Speed_1",
            "Rated_COP_at_Speed_2",
        ]

        all_object_types = idf.idfobjects.keys()

        cooling_equipment_types = [
            obj_type
            for obj_type in all_object_types
            if obj_type.startswith("COIL:COOLING") or obj_type.startswith("CHILLER:")
        ]

        for equipment_type in cooling_equipment_types:
            try:
                equipment_list = idf.idfobjects.get(equipment_type, [])

                for equipment in equipment_list:
                    for cop_field_name in cop_field_names:
                        if hasattr(equipment, cop_field_name):
                            setattr(equipment, cop_field_name, parameters.cooling_cop)
                            self._logger.debug(
                                f"Set {cop_field_name} to {parameters.cooling_cop} for {equipment.Name}"
                            )
                            modified_count += 1
            except Exception:
                self._logger.exception(f"Failed to process {equipment_type} objects")
                continue

        self._logger.info(f"Modified {modified_count} coil and chiller objects")

    def _apply_heating_coil_and_chiller_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply heating coil and chiller parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        modified_count = 0

        cop_field_names = [
            "Gross_Rated_Heating_COP",
            "Reference_COP",
            "Rated_COP",
            "High_Speed_Gross_Rated_Heating_COP",
            "Low_Speed_Gross_Rated_Heating_COP",
            "Rated_COP_at_Speed_1",
            "Rated_COP_at_Speed_2",
        ]

        all_object_types = idf.idfobjects.keys()

        heating_equipment_types = [
            obj_type
            for obj_type in all_object_types
            if obj_type.startswith("COIL:HEATING")
        ]

        for equipment_type in heating_equipment_types:
            try:
                equipment_list = idf.idfobjects.get(equipment_type, [])

                for equipment in equipment_list:
                    for cop_field_name in cop_field_names:
                        if hasattr(equipment, cop_field_name):
                            setattr(equipment, cop_field_name, parameters.heating_cop)
                            self._logger.debug(
                                f"Set {cop_field_name} to {parameters.heating_cop} for {equipment.Name}"
                            )
                            modified_count += 1
            except Exception:
                self._logger.exception(f"Failed to process {equipment_type} objects")
                continue

        self._logger.info(f"Modified {modified_count} heating coil objects")

    def _apply_cooling_air_temperature_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply cooling air temperature parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        sizing_zone_objects = idf.idfobjects.get("SIZING:ZONE", [])
        modified_count = 0

        for sizing_zone in sizing_zone_objects:
            sizing_zone.Zone_Cooling_Design_Supply_Air_Temperature = (
                parameters.cooling_air_temperature
            )
            self._logger.debug(
                f"Set cooling air temperature to {parameters.cooling_air_temperature}°C for {sizing_zone.Zone_or_ZoneList_Name}"
            )
            modified_count += 1

        self._logger.info(f"Modified {modified_count} sizing zone objects")

    def _apply_heating_air_temperature_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply heating air temperature parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        sizing_zone_objects = idf.idfobjects.get("SIZING:ZONE", [])
        modified_count = 0

        for sizing_zone in sizing_zone_objects:
            sizing_zone.Zone_Heating_Design_Supply_Air_Temperature = (
                parameters.heating_air_temperature
            )
            self._logger.debug(
                f"Set heating air temperature to {parameters.heating_air_temperature}°C for {sizing_zone.Zone_or_ZoneList_Name}"
            )
            modified_count += 1

        self._logger.info(f"Modified {modified_count} sizing zone objects")

    def _apply_lighting_parameters(
        self, context: SimulationContext, parameters: ECMParameters
    ) -> None:
        """
        apply lighting parameters to idf object

        Args:
            context (SimulationContext): Simulation context
            parameters (ECMParameters): ECM parameters
        """
        idf = context.idf

        lights = idf.idfobjects.get("LIGHTS", [])
        lighting_power_reduction = parameters.lighting_power_reduction
        modified_count = 0

        if not lights:
            self._logger.warning("No LIGHTS objects found in IDF")
            return

        if lighting_power_reduction is None:
            self._logger.warning("Lighting power reduction is not set")
            return

        for light in lights:
            calc_method = light.Design_Level_Calculation_Method

            if calc_method == "LightingLevel":
                original_level = light.Lighting_Level
                light.Lighting_Level = original_level * lighting_power_reduction
                modified_count += 1
            elif calc_method == "Watts/Area":
                original_power = light.Watts_per_Floor_Area
                light.Watts_per_Floor_Area = original_power * lighting_power_reduction
                modified_count += 1
            elif calc_method == "Watts/Person":
                original_power = light.Watts_per_Person
                light.Watts_per_Person = original_power * lighting_power_reduction
                modified_count += 1
            else:
                self._logger.warning(
                    f"Unsupported lighting calculation method: {calc_method}"
                )
                continue

        self._logger.info(f"Modified {modified_count} lighting objects")

    def _remove_objects(self, idf: IDF, object_type: str) -> None:
        objects = list(idf.idfobjects.get(object_type, []))
        for obj in objects:
            idf.removeidfobject(obj)
            self._logger.debug(f"Removed {object_type} object: {obj}")
