import sys
from pathlib import Path
from typing import NamedTuple

from idfpy import IDF
from idfpy.models import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Zone,
    ZoneInfiltrationDesignFlowRate,
    ZoneVentilationWindandStackOpenArea,
)
from joblib import Parallel, cpu_count, delayed
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from backend.utils.config.logger import set_logger

idf_paths = Path("backend/data/idfs/").glob("**/*.idf")


class NormalVector(NamedTuple):
    normal: tuple[float, float, float]
    area: float


def process_idf(idf_path: Path) -> None:
    set_logger()
    idf = IDF.load(idf_path)
    zones = idf.all_of_type(Zone)
    has_infiltration = any(idf.all_of_type(ZoneInfiltrationDesignFlowRate))
    has_open_area = any(idf.all_of_type(ZoneVentilationWindandStackOpenArea))
    if has_infiltration and has_open_area:
        logger.info("Skipping IDF with both infiltration and open area")
        return
    for zone in zones.values():
        zone_name = zone.name
        if not has_infiltration:
            infiltration = ZoneInfiltrationDesignFlowRate(
                name=f"{zone_name}_infiltration",
                zone_or_zonelist_or_space_or_spacelist_name=zone_name,
                design_flow_rate_calculation_method="AirChanges/Hour",
                design_flow_rate=0.0,
                flow_rate_per_floor_area=0.0,
                flow_rate_per_exterior_surface_area=0.0,
                air_changes_per_hour=0.0,
            )
            idf.add(infiltration)
        if not has_open_area:
            surfaces = zone.referencing(BuildingSurfaceDetailed)
            outdoor_surfaces = [
                surface
                for surface in surfaces
                if surface.outside_boundary_condition == "Outdoors"
            ]
            for surface in outdoor_surfaces:
                for fenestration_surface in surface.referencing(
                    FenestrationSurfaceDetailed
                ):
                    open_area = ZoneVentilationWindandStackOpenArea(
                        name=f"{zone_name}_{fenestration_surface.name}_open_area",
                        zone_or_space_name=zone_name,
                        opening_area=0.0,
                        effective_angle=fenestration_surface.azimuth,
                        height_difference=0.0,
                    )
                    idf.add(open_area)
    idf.save(idf_path)


if __name__ == "__main__":
    Parallel(n_jobs=cpu_count(), backend="loky")(
        delayed(process_idf)(idf_path) for idf_path in idf_paths
    )
    logger.info("Done")
