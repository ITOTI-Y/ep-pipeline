from typing import Final

from idfpy import IDF
from idfpy.models import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    SimulationControl,
    SizingPlant,
    Zone,
    ZoneInfiltrationDesignFlowRate,
    ZoneVentilationWindandStackOpenArea,
)
from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager

NATURAL_VENTILATION_MAX_OUTDOOR_C: Final = 26.0
NATURAL_VENTILATION_MIN_OUTDOOR_C: Final = 18.0
NATURAL_VENTILATION_MIN_INDOOR_C: Final = 18.0
NATURAL_VENTILATION_DELTA_C: Final = 1.0


class GeneralApply(IApply):
    def __init__(self, config: ConfigManager):
        self._config = config

    def apply(self, job: SimulationJob) -> None:
        logger.info("Applying setting configuration")
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        self._configure_setting(job.idf)
        self._add_infiltration(job.idf)
        self._add_natural_ventilation(job.idf)
        logger.success("Setting configuration applied successfully")

    def _configure_setting(self, idf: IDF) -> None:
        sim_control_list = idf.all_of_type(SimulationControl)
        has_sizing_plant = bool(idf.all_of_type(SizingPlant))

        for sim_control in sim_control_list.values():
            sim_control.do_zone_sizing_calculation = "Yes"
            sim_control.do_system_sizing_calculation = "Yes"
            sim_control.do_plant_sizing_calculation = (
                "Yes" if has_sizing_plant else "No"
            )
            sim_control.run_simulation_for_sizing_periods = "No"
            sim_control.run_simulation_for_weather_file_run_periods = "Yes"
            sim_control.do_hvac_sizing_simulation_for_sizing_periods = (
                "Yes" if has_sizing_plant else "No"
            )
            sim_control.maximum_number_of_hvac_sizing_simulation_passes = 2

    def _add_infiltration(self, idf: IDF) -> None:
        """Add a zero-ACH infiltration object to every zone that has none."""
        added = 0
        for zone in idf.all_of_type(Zone).values():
            if zone.referencing(ZoneInfiltrationDesignFlowRate):
                continue
            idf.add(
                ZoneInfiltrationDesignFlowRate(
                    name=f"{zone.name}_infiltration",
                    zone_or_zonelist_or_space_or_spacelist_name=zone.name,
                    design_flow_rate_calculation_method="AirChanges/Hour",
                    design_flow_rate=0.0,
                    flow_rate_per_floor_area=0.0,
                    flow_rate_per_exterior_surface_area=0.0,
                    air_changes_per_hour=0.0,
                )
            )
            added += 1
        logger.info(f"Added {added} zero-flow infiltration objects")

    def _add_natural_ventilation(self, idf: IDF) -> None:
        """Add a closed opening for every exterior window that has none.

        The object name embeds the window name; ECMApply relies on it to
        distribute the ventilation area over the zone's windows.
        """
        added = 0
        for zone in idf.all_of_type(Zone).values():
            for surface in zone.referencing(BuildingSurfaceDetailed):
                if surface.outside_boundary_condition != "Outdoors":
                    continue
                for window in surface.referencing(FenestrationSurfaceDetailed):
                    name = f"{zone.name}_{window.name}_open_area"
                    if idf.has(ZoneVentilationWindandStackOpenArea, name):
                        continue
                    idf.add(
                        ZoneVentilationWindandStackOpenArea(
                            name=name,
                            zone_or_space_name=zone.name,
                            opening_area=0.0,
                            effective_angle=window.azimuth,
                            height_difference=0.0,
                            maximum_outdoor_temperature=(
                                NATURAL_VENTILATION_MAX_OUTDOOR_C
                            ),
                            minimum_outdoor_temperature=(
                                NATURAL_VENTILATION_MIN_OUTDOOR_C
                            ),
                            minimum_indoor_temperature=NATURAL_VENTILATION_MIN_INDOOR_C,
                            delta_temperature=NATURAL_VENTILATION_DELTA_C,
                        )
                    )
                    added += 1
        logger.info(f"Added {added} closed natural ventilation openings")
