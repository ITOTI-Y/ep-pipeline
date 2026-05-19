import json
from pathlib import Path
from typing import Final

from shapely.geometry import shape

GEO_PROVINCE_FILE: Final = "province.geojson"
GEO_CITY_FILE: Final = "city.geojson"
GEO_COUNTY_FILE: Final = "county.geojson"

MAP_EXTENT: Final = [72, 137, 15, 55]  # [lon_min, lon_max, lat_min, lat_max]
SCS_EXTENT: Final = [104.5, 125, 0, 26]
CENTER_LONGITUDE: Final = (MAP_EXTENT[0] + MAP_EXTENT[1]) / 2

INSET_H = 0.40
INSET_W = (
    INSET_H
    * ((SCS_EXTENT[1] - SCS_EXTENT[0]) / (SCS_EXTENT[3] - SCS_EXTENT[2]))
    / ((MAP_EXTENT[1] - MAP_EXTENT[0]) / (MAP_EXTENT[3] - MAP_EXTENT[2]))
)


def load_geojson(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    return [shape(feat["geometry"]) for feat in data["features"]]
