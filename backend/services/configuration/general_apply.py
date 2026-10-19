from pathlib import Path
from typing import Final

from idfpy import IDF
from idfpy.models import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    FoundationKiva,
    SimulationControl,
    SiteGroundTemperatureBuildingSurface,
    SizingPlant,
    SurfacePropertyExposedFoundationPerimeter,
    Zone,
    ZoneInfiltrationDesignFlowRate,
    ZoneVentilationWindandStackOpenArea,
)
from loguru import logger

from backend.models import SimulationJob
from backend.services.configuration.ground_boundary import (
    COLLINEAR_TOLERANCE_M,
    Point,
    annual_mean_dry_bulb,
    bottom_edge,
    exposed_perimeter,
    floor_shared_length,
    horizontal_length,
)
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager

NATURAL_VENTILATION_MAX_OUTDOOR_C: Final = 26.0
NATURAL_VENTILATION_MIN_OUTDOOR_C: Final = 18.0
NATURAL_VENTILATION_MIN_INDOOR_C: Final = 18.0
NATURAL_VENTILATION_DELTA_C: Final = 1.0
FOUNDATION_WALL_SHARED_FRACTION: Final = 0.99
MONTHS: Final = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


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
        self._add_ground_foundations(job.idf, job.weather_file.file_path)
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
            if not any(
                surface.surface_type == "Wall"
                and surface.outside_boundary_condition == "Outdoors"
                for surface in zone.referencing(BuildingSurfaceDetailed)
            ):
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

    def _add_ground_foundations(self, idf: IDF, weather_path: Path) -> None:
        """Model ground-contact floors with Kiva and fix the remaining ground walls.

        Every floor facing the ground gets its own Foundation:Kiva object. A ground
        wall in the same zone joins that foundation when its bottom edge lies on
        the floor outline; walls without such a floor keep the Ground boundary
        and use the annual mean dry-bulb temperature of the weather file.
        """
        surfaces = list(idf.all_of_type(BuildingSurfaceDetailed).values())
        wall_bottoms = [
            bottom_edge(_points(surface))
            for surface in surfaces
            if surface.surface_type == "Wall"
            and surface.outside_boundary_condition in ("Outdoors", "Ground")
        ]
        ground_walls = [
            surface
            for surface in surfaces
            if surface.surface_type == "Wall"
            and surface.outside_boundary_condition == "Ground"
        ]
        floors = [
            surface
            for surface in surfaces
            if surface.surface_type == "Floor"
            and surface.outside_boundary_condition == "Ground"
        ]
        attached = 0
        for floor in floors:
            floor_points = _points(floor)
            floor_z = floor_points[0][2]
            foundation_name = f"{floor.name}_kiva"
            idf.add(FoundationKiva(name=foundation_name))
            wall_length = 0.0
            for wall in ground_walls:
                if (
                    wall.outside_boundary_condition != "Ground"
                    or wall.zone_name != floor.zone_name
                ):
                    continue
                edge = bottom_edge(_points(wall))
                if abs(edge[0][2] - floor_z) > COLLINEAR_TOLERANCE_M:
                    continue
                length = horizontal_length(edge)
                if (
                    length > 0.0
                    and floor_shared_length(floor_points, edge)
                    >= FOUNDATION_WALL_SHARED_FRACTION * length
                ):
                    wall.outside_boundary_condition = "Foundation"
                    wall.outside_boundary_condition_object = foundation_name
                    wall_length += length
                    attached += 1
            floor.outside_boundary_condition = "Foundation"
            floor.outside_boundary_condition_object = foundation_name
            idf.add(
                SurfacePropertyExposedFoundationPerimeter(
                    surface_name=floor.name,
                    exposed_perimeter_calculation_method="TotalExposedPerimeter",
                    total_exposed_perimeter=max(
                        exposed_perimeter(floor_points, wall_bottoms), wall_length
                    ),
                )
            )
        remaining = [
            wall for wall in ground_walls if wall.outside_boundary_condition == "Ground"
        ]
        if remaining:
            mean = annual_mean_dry_bulb(weather_path)
            idf.add(
                SiteGroundTemperatureBuildingSurface(
                    **{f"{month}_ground_temperature": mean for month in MONTHS}
                )
            )
        logger.info(
            f"Added {len(floors)} Kiva foundations with {attached} attached walls; "
            f"{len(remaining)} ground walls use the annual mean temperature"
        )


def _points(surface: BuildingSurfaceDetailed) -> list[Point]:
    if not surface.vertices:
        raise ValueError(f"Surface {surface.name} has no vertices")
    return [
        (
            vertex.vertex_x_coordinate,
            vertex.vertex_y_coordinate,
            vertex.vertex_z_coordinate,
        )
        for vertex in surface.vertices
    ]
