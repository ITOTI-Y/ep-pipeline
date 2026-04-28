from pydantic import BaseModel, ConfigDict, Field


class BaseCitySchema(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class PreprocessConfigSchema(BaseCitySchema):
    geo_weight: float = Field(
        0.3, ge=0.0, le=1.0, description="The weight of the geographic feature"
    )
    pca_variance: float = Field(
        0.95, gt=0.0, le=1.0, description="The variance of the PCA"
    )


class ClusterConfigSchema(BaseCitySchema):
    k_min: int = Field(20, ge=2, description="The minimum number of clusters")
    k_max: int = Field(60, ge=2, description="The maximum number of clusters")
    override_k: int | None = Field(
        None, ge=2, description="The number of clusters to override"
    )
    n_gap_refs: int = Field(20, ge=5, description="The number of gap references")


class DownloadConfigSchema(BaseCitySchema):
    request_interval: float = Field(1.5, ge=0.0)
    max_retries: int = Field(3, ge=1)
    retry_wait: float = Field(60.0, ge=1.0)
    backoff_wait: float = Field(30.0, ge=1.0)
    concurrency: int = Field(5, ge=1)


class CitySelectionConfigSchema(BaseCitySchema):
    forced_cities: list[str] = Field(
        default_factory=list, description="The forced cities"
    )
    preprocess: PreprocessConfigSchema = PreprocessConfigSchema()
    cluster: ClusterConfigSchema = ClusterConfigSchema()
    download: DownloadConfigSchema = DownloadConfigSchema()
