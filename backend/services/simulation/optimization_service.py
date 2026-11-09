import json

from joblib import load
from loguru import logger
from sklearn.preprocessing import OneHotEncoder

from backend._share import TARGET_NAMES
from backend.models import SimulationJob, SimulationResult
from backend.services.configuration import (
    ECMApply,
    GeneralApply,
    OutputApply,
    PeriodApply,
)
from backend.services.interfaces import (
    IEnergyPlusExecutor,
    IResultParser,
)
from backend.services.optimization.optimization_model import GeneticAlgorithmModel
from backend.services.optimization.surrogate_model import ISurrogateModel
from backend.services.simulation._share import (
    IDataExtractor,
    IFileCleaner,
    ISimulationService,
    job_surrogate_model_path,
)
from backend.utils.config import ConfigManager


class OptimizationService(ISimulationService):
    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: IResultParser,
        file_cleaner: IFileCleaner,
        data_extractor: IDataExtractor,
        config: ConfigManager,
        job: SimulationJob,
    ):
        super().__init__(
            executor, result_parser, file_cleaner, data_extractor, config, job
        )
        self._surrogate_model, self._one_hot_encoder = self._load_models()
        self._ecm_apply = ECMApply()
        self._output_apply = OutputApply(config=config)
        self._period_apply = PeriodApply(config=config)
        self._general_apply = GeneralApply(config=config)

        self._predicted_eui: float | None = None

    def _load_models(self) -> tuple[ISurrogateModel, OneHotEncoder]:
        surrogate_model_path = job_surrogate_model_path(
            self._config, self._job.idf_file.city, self._job.idf_file.building_type
        )
        encode_model_path = self._config.paths.sim_dir / "encode_model.pkl"
        for path in [surrogate_model_path, encode_model_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required model not found: {path}")
        logger.info(f"Surrogate model loaded from {surrogate_model_path}")
        return load(surrogate_model_path), load(encode_model_path)

    def _surrogate_rmse(self) -> float:
        evaluate_path = (
            job_surrogate_model_path(
                self._config, self._job.idf_file.city, self._job.idf_file.building_type
            ).parent
            / "evaluate.json"
        )
        target_index = TARGET_NAMES.index("total_site_eui")
        return json.loads(evaluate_path.read_text())[f"output_{target_index + 1}_rmse"]

    def _get_best_ecm_parameters(self) -> None:
        optimization_model = GeneticAlgorithmModel(
            config=self._config,
            surrogate_model=self._surrogate_model,
            encode_model=self._one_hot_encoder,
            code=self._job.weather_file.code,
        )
        best_ecm, predicted_eui = optimization_model.optimize(
            building_type=self._job.idf_file.building_type
        )
        relaxation = self._config.optimization.relaxation
        best_ecm, predicted_eui = optimization_model.relax_to_reference(
            best_ecm=best_ecm,
            best_fitness=predicted_eui,
            reference_values=relaxation.reference_values,
            tolerance=relaxation.rmse_multiplier * self._surrogate_rmse(),
        )
        self._predicted_eui = predicted_eui
        self._job = self._job.model_copy(update={"ecm_parameters": best_ecm})

    def prepare(self) -> None:
        self._get_best_ecm_parameters()
        self._output_apply.apply(self._job)
        self._period_apply.apply(self._job)
        self._general_apply.apply(self._job)
        self._ecm_apply.apply(self._job)
        logger.info("Optimization preparation completed")

    def cleanup(self) -> None:
        self._file_cleaner.clean(
            job=self._job,
            config=self._config,
            exclude_files=self._cleanup_exclude,
        )

    def run(self) -> SimulationResult:
        result = super().run()
        result.ecm_parameters = self._job.ecm_parameters or None
        result.predicted_eui = self._predicted_eui
        result.weather_code = self._job.weather_file.code
        return result
