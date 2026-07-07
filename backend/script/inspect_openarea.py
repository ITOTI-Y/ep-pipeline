from pathlib import Path
from typing import NamedTuple, cast

from idfpy import IDF
from idfpy.models import (
    BuildingSurfaceDetailed,
    Zone,
    ZoneVentilationWindandStackOpenArea,
)
from joblib import Parallel, cpu_count, delayed

idf_paths = Path("backend/data/idfs/").glob("**/*.idf")


class NormalVector(NamedTuple):
    normal: tuple[float, float, float]
    area: float


def process_idf(idf_path: Path) -> None:
    idf = IDF.load(idf_path)
    zones = idf.all_of_type(Zone)
    for zone in zones.values():
        zone_name = zone.name
        surfaces = cast(
            list[BuildingSurfaceDetailed], zone.referencing(BuildingSurfaceDetailed)
        )
        outdoor_surfaces = [
            surface
            for surface in surfaces
            if surface.outside_boundary_condition == "Outdoors"
        ]
        for surface in outdoor_surfaces:
            open_area = ZoneVentilationWindandStackOpenArea(
                name=f"{zone_name}_{surface.name}_open_area",
                zone_or_space_name=zone_name,
                opening_area=0.0,
                effective_angle=surface.azimuth,
                height_difference=0.0,
            )
            idf.add(open_area)
    idf.save(idf_path)


Parallel(n_jobs=cpu_count(), backend="loky")(
    delayed(process_idf)(idf_path) for idf_path in idf_paths
)
