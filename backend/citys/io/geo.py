import json
from functools import lru_cache
from pathlib import Path
from typing import Final, NamedTuple

from loguru import logger
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import PreparedGeometry, prep

GB_PREFIX_TO_ISO: Final[dict[int, str]] = {
    11: "BJ",
    12: "TJ",
    13: "HE",
    14: "SX",
    15: "NM",
    21: "LN",
    22: "JL",
    23: "HL",
    31: "SH",
    32: "JS",
    33: "ZJ",
    34: "AH",
    35: "FJ",
    36: "JX",
    37: "SD",
    41: "HA",
    42: "HB",
    43: "HN",
    44: "GD",
    45: "GX",
    46: "HI",
    50: "CQ",
    51: "SC",
    52: "GZ",
    53: "YN",
    54: "XZ",
    61: "SN",
    62: "GS",
    63: "QH",
    64: "NX",
    65: "XJ",
    71: "TW",
    81: "HK",
    82: "MO",
}
_FALLBACK_WARN_DEG: Final = 0.5


class Entries(NamedTuple):
    code: str
    shp: BaseGeometry
    prepared_shp: PreparedGeometry


class ProvinceLocator:
    def __init__(self, geojson_path: Path):
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

        data = json.loads(geojson_path.read_text(encoding="utf-8"))
        self._entries: list[Entries] = []
        for feat in data["features"]:
            geom = feat.get("geometry")
            gb = str(feat.get("properties", {}).get("gb", ""))
            if geom is None or len(gb) < 5:
                continue
            code = GB_PREFIX_TO_ISO.get(int(gb[3:5]))
            if code is None:
                logger.warning(f"Unknown Province GB code: {gb}")
                continue
            shp = shape(geom)
            self._entries.append(Entries(code, shp, prep(shp)))
        if not self._entries:
            raise ValueError(f"No usable province polygons in {geojson_path}")

    def lookup(self, latitude: float, longitude: float) -> str:
        point = Point(longitude, latitude)
        for code, _, prepared_shp in self._entries:
            if prepared_shp.contains(point):
                return code
        code, shp, _ = min(self._entries, key=lambda e: e.shp.distance(point))
        dist = shp.distance(point)
        if dist > _FALLBACK_WARN_DEG:
            logger.warning(
                f"({latitude}, {longitude}) outside all province polygons; "
                f"assigned nearest province {code} at {dist:.2f} deg"
            )
        return code


@lru_cache(maxsize=1)
def _default_locator() -> ProvinceLocator:
    from backend.utils.config import ConfigManager

    geo_dir = Path(ConfigManager().paths.geo_dir)
    return ProvinceLocator(geo_dir / "province.geojson")


def lookup_province(latitude: float, longitude: float) -> str:
    """Resolve a province code using the process-wide cached locator."""
    return _default_locator().lookup(latitude, longitude)
