from idfpy.models import (
    AirTerminalSingleDuctVAVReheat,
    BuildingSurfaceDetailed,
    ChillerElectricReformulatedEIR,
    Construction,
    CoolingTowerVariableSpeed,
    FenestrationSurfaceDetailed,
    Lights,
    MaterialNoMass,
    ScheduleConstant,
    SizingZone,
    SurfaceControlMovableInsulation,
    WindowMaterialSimpleGlazingSystem,
    ZoneInfiltrationDesignFlowRate,
    ZoneVentilationWindandStackOpenArea,
)
from loguru import logger

from backend.models import ECMParameters, SimulationJob
from backend.services.configuration.iapply import IApply


class ECMApply(IApply):
    def __init__(self):
        super().__init__()

    def apply(self, job: SimulationJob) -> None:
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
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply windows parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
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
            idf.all_of_type(WindowMaterialSimpleGlazingSystem).get(window_material_name)
            is not None
        ):
            return

        idf.add(
            WindowMaterialSimpleGlazingSystem(
                name=window_material_name,
                u_factor=parameters.window_u_value,
                solar_heat_gain_coefficient=parameters.window_shgc,
                visible_transmittance=parameters.visible_transmittance,
            )
        )

        constructions_name = f"Construction_window_{window_material_name}"

        idf.add(
            Construction(
                name=constructions_name,
                outside_layer=window_material_name,
            )
        )

        fenestration_surfaces = idf.all_of_type(FenestrationSurfaceDetailed)
        modified_count = 0
        for surface_name, surface in fenestration_surfaces.items():
            if surface.surface_type.upper() == "WINDOW":
                surface.construction_name = constructions_name
                logger.debug(
                    f"Set construction name to {constructions_name} for {surface_name}"
                )
                modified_count += 1

        logger.info(f"Modified {modified_count} fenestration surface objects")

    def _apply_wall_insulation_parameters(
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply wall insulation parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
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

        idf.add(
            MaterialNoMass(
                name=insulation_materials_name,
                roughness="Rough",
                thermal_resistance=parameters.wall_insulation,
                thermal_absorptance=0.9,
                solar_absorptance=0.7,
                visible_absorptance=0.7,
            )
        )

        schedule_name = "WallInsulationSchedule_AlwaysOn"
        if idf.all_of_type(ScheduleConstant).get(schedule_name) is None:
            idf.add(
                ScheduleConstant(
                    name=schedule_name,
                    hourly_value=1.0,
                )
            )

        self._remove_objects(idf, SurfaceControlMovableInsulation)

        surfaces = idf.all_of_type(BuildingSurfaceDetailed)
        modified_count = 0
        for surface in surfaces.values():
            if (
                surface.outside_boundary_condition == "Outdoors"
                and surface.surface_type.upper() in ["WALL", "ROOF"]
            ):
                idf.add(
                    SurfaceControlMovableInsulation(
                        insulation_type="Outside",
                        surface_name=surface.name,
                        material_name=insulation_materials_name,
                        schedule_name=schedule_name,
                    )
                )
                logger.debug(
                    f"Set insulation material to {insulation_materials_name} for {surface.name}"
                )
                modified_count += 1

        logger.info(f"Modified {modified_count} surface control objects")

    def _apply_infiltration_parameters(
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply infiltration parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
        """
        if parameters.infiltration_rate is None:
            logger.warning("Infiltration rate is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        infiltration_objects = idf.all_of_type(ZoneInfiltrationDesignFlowRate)
        modified_count = 0

        if not infiltration_objects:
            logger.warning("No ZoneInfiltrationDesignFlowRate objects found in IDF")
            return

        for infiltration in infiltration_objects.values():
            infiltration.design_flow_rate_calculation_method = "AirChanges/Hour"
            infiltration.air_changes_per_hour = parameters.infiltration_rate
            logger.debug(
                f"Set infiltration rate to {parameters.infiltration_rate} ACH for {infiltration.name}"
            )
            modified_count += 1

        logger.info(f"Modified {modified_count} infiltration objects")

    def _apply_natural_ventilation_parameters(
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply natural ventilation parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
        """
        if parameters.natural_ventilation_area is None:
            logger.warning("Natural ventilation area is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        zone_ventilations = idf.all_of_type(ZoneVentilationWindandStackOpenArea)
        modified_count = 0

        for zone_ventilation in zone_ventilations.values():
            zone_ventilation.opening_area = parameters.natural_ventilation_area
            logger.debug(
                f"Set natural ventilation area to {parameters.natural_ventilation_area} m² for {zone_ventilation.name}"
            )
            modified_count += 1

        logger.info(f"Modified {modified_count} ventilation objects")

    def _apply_cooling_coil_and_chiller_parameters(
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply cooling coil and chiller parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
        """
        if parameters.cooling_cop is None:
            logger.warning("Cooling COP is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

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

        all_object_types = idf.types()

        cooling_equipment_types = [
            obj_type
            for obj_type in all_object_types
            if obj_type.upper().startswith("COIL:COOLING")
            or obj_type.upper().startswith("CHILLER:")
        ]

        for equipment_type in cooling_equipment_types:
            try:
                equipment_list = idf.all_of_type(equipment_type)

                for equipment in equipment_list.values():
                    for cop_field_name in cop_field_names:
                        if hasattr(equipment, cop_field_name):
                            setattr(equipment, cop_field_name, parameters.cooling_cop)
                            logger.debug(
                                f"Set {cop_field_name} to {parameters.cooling_cop} for {equipment.name}"  # type: ignore
                            )
                            modified_count += 1
            except Exception:
                logger.exception(f"Failed to process {equipment_type} objects")
                continue

        logger.info(f"Modified {modified_count} coil and chiller objects")

    def _apply_heating_coil_and_chiller_parameters(
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply heating coil and chiller parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
        """
        if parameters.heating_cop is None:
            logger.warning("Heating COP is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

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

        all_object_types = idf.types()

        heating_equipment_types = [
            obj_type
            for obj_type in all_object_types
            if obj_type.upper().startswith("COIL:HEATING")
        ]

        for equipment_type in heating_equipment_types:
            try:
                equipment_list = idf.all_of_type(equipment_type)

                for equipment in equipment_list.values():
                    for cop_field_name in cop_field_names:
                        if hasattr(equipment, cop_field_name):
                            setattr(equipment, cop_field_name, parameters.heating_cop)
                            logger.debug(
                                f"Set {cop_field_name} to {parameters.heating_cop} for {equipment.name}"  # type: ignore
                            )
                            modified_count += 1
            except Exception:
                logger.exception(f"Failed to process {equipment_type} objects")
                continue

        logger.info(f"Modified {modified_count} heating coil objects")

    def _apply_cooling_air_temperature_parameters(
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply cooling air temperature parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
        """
        if parameters.cooling_air_temperature is None:
            logger.warning("Cooling air temperature is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        sizing_zone_objects = idf.all_of_type(SizingZone)
        modified_count = 0

        for sizing_zone in sizing_zone_objects.values():
            sizing_zone.zone_cooling_design_supply_air_temperature = (
                parameters.cooling_air_temperature
            )
            logger.debug(
                f"Set cooling air temperature to {parameters.cooling_air_temperature}°C for {sizing_zone.zone_or_zonelist_name}"
            )
            modified_count += 1

        logger.info(f"Modified {modified_count} sizing zone objects")

    def _apply_heating_air_temperature_parameters(
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply heating air temperature parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
        """
        if parameters.heating_air_temperature is None:
            logger.warning("Heating air temperature is not set, skipping")
            return

        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        sizing_zone_objects = idf.all_of_type(SizingZone)
        modified_count = 0

        for sizing_zone in sizing_zone_objects.values():
            sizing_zone.zone_heating_design_supply_air_temperature = (
                parameters.heating_air_temperature
            )
            logger.debug(
                f"Set heating air temperature to {parameters.heating_air_temperature}°C for {sizing_zone.zone_or_zonelist_name}"
            )
            modified_count += 1

        logger.info(f"Modified {modified_count} sizing zone objects")

    def _apply_lighting_parameters(
        self, job: SimulationJob, parameters: ECMParameters
    ) -> None:
        """
        apply lighting parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
        """
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        lights = idf.all_of_type(Lights)
        lighting_power_reduction = parameters.lighting_power_reduction
        modified_count = 0

        if not lights:
            logger.warning("No LIGHTS objects found in IDF")
            return

        if lighting_power_reduction is None:
            logger.warning("Lighting power reduction is not set")
            return

        for light in lights.values():
            calc_method = light.design_level_calculation_method

            if calc_method == "LightingLevel":
                original_level = light.lighting_level
                if original_level is None:
                    logger.warning(
                        f"Lighting level is not set for {light.name}, because the design level calculation method is not set"
                    )
                    continue
                light.lighting_level = original_level * (1 - lighting_power_reduction)
                modified_count += 1
            elif calc_method == "Watts/Area":
                original_power = light.watts_per_floor_area
                if original_power is None:
                    logger.warning(
                        f"Watts per floor area is not set for {light.name}, because the design level calculation method is not set"
                    )
                    continue
                light.watts_per_floor_area = original_power * (
                    1 - lighting_power_reduction
                )
                modified_count += 1
            elif calc_method == "Watts/Person":
                original_power = light.watts_per_person
                if original_power is None:
                    logger.warning(
                        f"Watts per person is not set for {light.name}, because the design level calculation method is not set"
                    )
                    continue
                light.watts_per_person = original_power * (1 - lighting_power_reduction)
                modified_count += 1
            else:
                logger.warning(
                    f"Unsupported lighting calculation method: {calc_method}"
                )
                continue

        logger.info(f"Modified {modified_count} lighting objects")

    def _apply_hvac_settings_parameters(self, job: SimulationJob) -> None:
        """
        apply hvac settings parameters to idf object

        Args:
            job (SimulationJob): Simulation job
            parameters (ECMParameters): ECM parameters
        """
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        idf = job.idf

        modified_count = 0

        vav_reheat_terminals = idf.all_of_type(AirTerminalSingleDuctVAVReheat)

        for terminal in vav_reheat_terminals.values():
            terminal.maximum_air_flow_rate = "Autosize"
            terminal.maximum_hot_water_or_steam_flow_rate = "Autosize"
            terminal.maximum_flow_fraction_during_reheat = "Autosize"
            terminal.constant_minimum_air_flow_fraction = "Autosize"
            terminal.fixed_minimum_air_flow_rate = "Autosize"
            modified_count += 1

        logger.info(f"Modified {modified_count} VAV reheat terminals")

        modified_count = 0

        chillers = idf.all_of_type(ChillerElectricReformulatedEIR)

        for chiller in chillers.values():
            chiller.reference_capacity = "Autosize"
            chiller.reference_chilled_water_flow_rate = "Autosize"
            chiller.reference_condenser_water_flow_rate = "Autosize"
            modified_count += 1

        logger.info(f"Modified {modified_count} chillers")

        modified_count = 0

        cooling_towers = idf.all_of_type(CoolingTowerVariableSpeed)
        for tower in cooling_towers.values():
            tower.design_water_flow_rate = "Autosize"
            tower.design_air_flow_rate = "Autosize"
            tower.design_fan_power = "Autosize"
            modified_count += 1

        logger.info(f"Modified {modified_count} CoolingTowers")
