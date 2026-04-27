from typing import Final

GROUP_A_COLS: Final = [
    "hdd18",
    "cdd18",
    "annual_mean_dew_point",
    "annual_ghi",
    "annual_dhi",
    "annual_mean_wind_speed",
]

GROUP_B_COLS: Final = [f"temp_m{i:02d}" for i in range(1, 13)] + [
    f"ghi_m{i:02d}" for i in range(1, 13)
]

GROUP_C_COLS: Final = ["latitude", "longitude", "elevation"]
META_COLS: Final = [
    "city_name",
    "province",
    "wmo_id",
    "latitude",
    "longitude",
    "elevation",
]
FEATURE_COLS: Final = GROUP_A_COLS + GROUP_B_COLS + GROUP_C_COLS

EARTH_RADIUS_KM: Final = 6371.0
