from typing import Final

TMYX_BASE_URL: Final = "https://climate.onebuilding.org/WMO_Region_2_Asia/CHN_China"
DEST_API_URL: Final = "https://svr.dest.net.cn/api/v1"
DEST_CATALOG_URL: Final = f"{DEST_API_URL}/all_model_names"
DEST_LOAD_URL: Final = f"{DEST_API_URL}/load_model_file"

BUILDING_TYPES: Final = [
    "Commercial office A",
    "Commercial office B",
    "Government office A",
    "Government office B",
    "High-rise apartment(slab type)",
    "High-rise apartment(tower type)",
    "Low-rise apartment",
    "Terraced house",
    "Large hotel",
    "Small hotel",
    "Shopping mall",
    "Primary/secondary school",
    "University",
    "Inpatient",
    "Outpatient",
]

BTYPE_SHORT: Final = {
    "Commercial office A": "CoA",
    "Commercial office B": "CoB",
    "Government office A": "GoA",
    "Government office B": "GoB",
    "High-rise apartment(slab type)": "HighS",
    "High-rise apartment(tower type)": "HighT",
    "Low-rise apartment": "Low",
    "Terraced house": "Th",
    "Large hotel": "LH",
    "Small hotel": "SH",
    "Shopping mall": "Mall",
    "Primary/secondary school": "Sch",
    "University": "Uni",
    "Inpatient": "Inp",
    "Outpatient": "Outp",
}

DEST_YEARS: Final = list(range(2019, 2000, -1))
