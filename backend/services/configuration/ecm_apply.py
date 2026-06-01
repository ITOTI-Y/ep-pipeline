from loguru import logger

from backend.models import ECMParametersSchema, SimulationJobSchema
from backend.services.configuration.iapply import IApply


class ECMApply(IApply):
    def __init__(self):
        super().__init__()

    def apply(self, job: SimulationJobSchema) -> None:
        logger.info("Applying ECM configuration")
        if job.ecm_parameters is None:
            logger.error("ECM parameters are not set, skipping")
            raise ValueError("ECM parameters are not set")
        parameters = job.ecm_parameters

        self._apply_window_parameters(job, parameters)
        self._apply_wall_insulation_parameters(job, parameters)
        self._apply_infiltration_parameters(job, parameters)
        self._apply_natural_ventilation_parameters(job, parameters)
        self._apply_cooling_coil_and_chiller_parameters(job, parameters)
        self._apply_heating_coil_and_chiller_parameters(job, parameters)
        self._apply_cooling_air_temperature_parameters(job, parameters)
        self._apply_heating_air_temperature_parameters(job, parameters)
        self._apply_lighting_parameters(job, parameters)
        self._apply_hvac_settings_parameters(job)
        logger.info("ECM configuration applied successfully")

    def _apply_window_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        """
        apply windows parameters to idf object

        Args:
            job (SimulationJobSchema): Simulation job
            parameters (ECMParametersSchema): ECM parameters
        """
        if (
            parameters.window_u_value is None
            or parameters.window_shgc is None
            or parameters.visible_transmittance is None
        ):
            logger.warning("Window parameters are not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

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
                logger.debug(
                    f"Set construction name to {constructions_name} for {surface.Name}"
                )
                modified_count += 1

        logger.info(f"Modified {modified_count} fenestration surface objects")

    def _apply_wall_insulation_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        """
        apply wall insulation parameters to idf object

        Args:
            job (SimulationJobSchema): Simulation job
            parameters (ECMParametersSchema): ECM parameters
        """
        if parameters.wall_insulation is None:
            logger.warning("Wall insulation is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

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
        if idf.getobject("SCHEDULE:CONSTANT", schedule_name) is None:
            idf.newidfobject(
                "Schedule:Constant",
                Name=schedule_name,
                Hourly_Value=1.0,
            )

        self._remove_objects(idf, "SurfaceControl:MovableInsulation")
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
                logger.debug(
                    f"Set insulation material to {insulation_materials_name} for {surface.Name}"
                )
                modified_count += 1

        logger.info(f"Modified {modified_count} surface control objects")

    def _apply_infiltration_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        """
        apply infiltration parameters to idf object

        Args:
            job (SimulationJobSchema): Simulation job
            parameters (ECMParametersSchema): ECM parameters
        """
        if parameters.infiltration_rate is None:
            logger.warning("Infiltration rate is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        infiltration_objects = idf.idfobjects.get("ZONEINFILTRATION:DESIGNFLOWRATE", [])
        modified_count = 0

        if not infiltration_objects:
            logger.warning("No ZONEINFILTRATION:DESIGNFLOWRATE objects found in IDF")
            return

        for infiltration in infiltration_objects:
            infiltration.Design_Flow_Rate_Calculation_Method = "AirChanges/Hour"
            infiltration.Air_Changes_per_Hour = parameters.infiltration_rate
            logger.debug(
                f"Set infiltration rate to {parameters.infiltration_rate} ACH for {infiltration.Name}"
            )
            modified_count += 1

        logger.info(f"Modified {modified_count} infiltration objects")

    def _apply_natural_ventilation_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        """
        apply natural ventilation parameters to idf object

        Args:
            job (SimulationJobSchema): Simulation job
            parameters (ECMParametersSchema): ECM parameters
        """
        if parameters.natural_ventilation_area is None:
            logger.warning("Natural ventilation area is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        zone_ventilations = idf.idfobjects.get(
            "ZONEVENTILATION:WindandStackOpenArea", []
        )
        modified_count = 0

        for zone_ventilation in zone_ventilations:
            zone_ventilation.Opening_Area = parameters.natural_ventilation_area
            logger.debug(
                f"Set natural ventilation area to {parameters.natural_ventilation_area} m² for {zone_ventilation.Name}"
            )
            modified_count += 1

        logger.info(f"Modified {modified_count} ventilation objects")

    def _apply_cop_parameters(
        self,
        job: SimulationJobSchema,
        cop_value: float | None,
        cop_field_names: list[str],
        equipment_type_prefixes: tuple[str, ...],
        label: str,
    ) -> None:
        """Write a single COP value into every matching coil/chiller field.

        Args:
            job: Simulation job holding the IDF.
            cop_value: COP to write; skipped when None.
            cop_field_names: IDF field names to set on each matching object.
            equipment_type_prefixes: Object-type prefixes selecting equipment.
            label: Human-readable tag for log messages.
        """
        if cop_value is None:
            logger.warning(f"{label} COP is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        modified_count = 0
        equipment_types = [
            obj_type
            for obj_type in idf.idfobjects
            if obj_type.startswith(equipment_type_prefixes)
        ]

        for equipment_type in equipment_types:
            try:
                for equipment in idf.idfobjects.get(equipment_type, []):
                    for cop_field_name in cop_field_names:
                        if hasattr(equipment, cop_field_name):
                            setattr(equipment, cop_field_name, cop_value)
                            logger.debug(
                                f"Set {cop_field_name} to {cop_value} for {equipment.Name}"
                            )
                            modified_count += 1
            except Exception:
                logger.exception(f"Failed to process {equipment_type} objects")
                continue

        logger.info(f"Modified {modified_count} {label} coil and chiller objects")

    def _apply_cooling_coil_and_chiller_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        self._apply_cop_parameters(
            job,
            parameters.cooling_cop,
            [
                "Gross_Rated_Cooling_COP",
                "Reference_COP",
                "Rated_COP",
                "High_Speed_Gross_Rated_Cooling_COP",
                "Low_Speed_Gross_Rated_Cooling_COP",
                "Rated_COP_at_Speed_1",
                "Rated_COP_at_Speed_2",
            ],
            ("COIL:COOLING", "CHILLER:"),
            "cooling",
        )

    def _apply_heating_coil_and_chiller_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        self._apply_cop_parameters(
            job,
            parameters.heating_cop,
            [
                "Gross_Rated_Heating_COP",
                "Reference_COP",
                "Rated_COP",
                "High_Speed_Gross_Rated_Heating_COP",
                "Low_Speed_Gross_Rated_Heating_COP",
                "Rated_COP_at_Speed_1",
                "Rated_COP_at_Speed_2",
            ],
            ("COIL:HEATING",),
            "heating",
        )

    def _apply_cooling_air_temperature_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        """
        apply cooling air temperature parameters to idf object

        Args:
            job (SimulationJobSchema): Simulation job
            parameters (ECMParametersSchema): ECM parameters
        """
        if parameters.cooling_air_temperature is None:
            logger.warning("Cooling air temperature is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        sizing_zone_objects = idf.idfobjects.get("SIZING:ZONE", [])
        modified_count = 0

        for sizing_zone in sizing_zone_objects:
            sizing_zone.Zone_Cooling_Design_Supply_Air_Temperature = (
                parameters.cooling_air_temperature
            )
            logger.debug(
                f"Set cooling air temperature to {parameters.cooling_air_temperature}°C for {sizing_zone.Zone_or_ZoneList_Name}"
            )
            modified_count += 1

        logger.info(f"Modified {modified_count} sizing zone objects")

    def _apply_heating_air_temperature_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        """
        apply heating air temperature parameters to idf object

        Args:
            job (SimulationJobSchema): Simulation job
            parameters (ECMParametersSchema): ECM parameters
        """
        if parameters.heating_air_temperature is None:
            logger.warning("Heating air temperature is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        sizing_zone_objects = idf.idfobjects.get("SIZING:ZONE", [])
        modified_count = 0

        for sizing_zone in sizing_zone_objects:
            sizing_zone.Zone_Heating_Design_Supply_Air_Temperature = (
                parameters.heating_air_temperature
            )
            logger.debug(
                f"Set heating air temperature to {parameters.heating_air_temperature}°C for {sizing_zone.Zone_or_ZoneList_Name}"
            )
            modified_count += 1

        logger.info(f"Modified {modified_count} sizing zone objects")

    def _apply_lighting_parameters(
        self, job: SimulationJobSchema, parameters: ECMParametersSchema
    ) -> None:
        """
        apply lighting parameters to idf object

        Args:
            job (SimulationJobSchema): Simulation job
            parameters (ECMParametersSchema): ECM parameters
        """
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        lights = idf.idfobjects.get("LIGHTS", [])
        lighting_power_reduction = parameters.lighting_power_reduction
        modified_count = 0

        if not lights:
            logger.warning("No LIGHTS objects found in IDF")
            return

        if lighting_power_reduction is None:
            logger.warning("Lighting power reduction is not set")
            return

        for light in lights:
            calc_method = light.Design_Level_Calculation_Method

            if calc_method == "LightingLevel":
                original_level = light.Lighting_Level
                light.Lighting_Level = original_level * (1 - lighting_power_reduction)
                modified_count += 1
            elif calc_method == "Watts/Area":
                original_power = light.Watts_per_Floor_Area
                light.Watts_per_Floor_Area = original_power * (
                    1 - lighting_power_reduction
                )
                modified_count += 1
            elif calc_method == "Watts/Person":
                original_power = light.Watts_per_Person
                light.Watts_per_Person = original_power * (1 - lighting_power_reduction)
                modified_count += 1
            else:
                logger.warning(
                    f"Unsupported lighting calculation method: {calc_method}"
                )
                continue

        logger.info(f"Modified {modified_count} lighting objects")

    def _apply_hvac_settings_parameters(self, job: SimulationJobSchema) -> None:
        """
        apply hvac settings parameters to idf object

        Args:
            job (SimulationJobSchema): Simulation job
            parameters (ECMParametersSchema): ECM parameters
        """
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        modified_count = 0

        vav_reheat_terminals = idf.idfobjects.get(
            "AIRTERMINAL:SINGLEDUCT:VAV:REHEAT", []
        )

        for terminal in vav_reheat_terminals:
            terminal.Maximum_Air_Flow_Rate = "AUTOSIZE"
            terminal.Maximum_Hot_Water_or_Steam_Flow_Rate = "AUTOSIZE"
            terminal.Maximum_Flow_Fraction_During_Reheat = "AUTOSIZE"
            terminal.Constant_Minimum_Air_Flow_Fraction = "AUTOSIZE"
            terminal.Fixed_Minimum_Air_Flow_Rate = "AUTOSIZE"
            modified_count += 1

        logger.info(f"Modified {modified_count} VAV reheat terminals")

        modified_count = 0

        chillers = idf.idfobjects.get("CHILLER:ELECTRIC:REFORMULATEDEIR", [])

        for chiller in chillers:
            chiller.Reference_Capacity = "AUTOSIZE"
            chiller.Reference_Chilled_Water_Flow_Rate = "AUTOSIZE"
            chiller.Reference_Condenser_Water_Flow_Rate = "AUTOSIZE"
            modified_count += 1

        logger.info(f"Modified {modified_count} chillers")

        modified_count = 0

        cooling_towers = idf.idfobjects.get("COOLINGTOWER:VARIABLESPEED", [])
        for tower in cooling_towers:
            tower.Design_Water_Flow_Rate = "AUTOSIZE"
            tower.Design_Air_Flow_Rate = "AUTOSIZE"
            tower.Design_Fan_Power = "AUTOSIZE"
            modified_count += 1

        logger.info(f"Modified {modified_count} CoolingTowers")
