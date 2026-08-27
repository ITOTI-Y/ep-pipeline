from backend.models.config_models import ECMParametersConfig

SSP_ORDER = {
    "tmy": 0,
    "ssp126": 1,
    "ssp245": 2,
    "ssp370": 3,
    "ssp434": 4,
    "ssp585": 5,
}
FEATURE_NAMES = ECMParametersConfig().keys

TARGET_NAMES = [
    "net_site_eui",
    "net_source_eui",
    "total_site_eui",
    "total_source_eui",
]
