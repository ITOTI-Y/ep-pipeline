from pathlib import Path
from pickle import dump, load

from eppy.modeleditor import IDF
from joblib import Parallel, delayed

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.models import (
    Building,
    BuildingType,
    ECMContext,
    SimulationJob,
    Weather,
)
from backend.models.enums import SimulationType
from backend.services.optimization import ParameterSampler
from backend.services.simulation import ECMService, FileCleaner, ResultParser
from backend.utils.config import ConfigManager


def test_batch_simulation():
    config = ConfigManager(Path("backend/configs"))

    idf_file_path = config.paths.idf_files[0]

    building = Building(
        name=idf_file_path.stem,
        building_type=BuildingType.OFFICE_LARGE,
        location="Chicago",
        idf_file_path=idf_file_path,
    )

    weather = Weather(
        file_path=config.paths.ftmy_files[0],
        location="Chicago",
    )

    output_directory = config.paths.ecm_dir / building.name

    n_samples = 10

    sampler = ParameterSampler(config=config)
    ecm_samples = sampler.sample(
        n_samples=n_samples, building_type=BuildingType.OFFICE_LARGE
    )

    jobs = []
    for i, ecm_parameters in enumerate(ecm_samples):
        job = SimulationJob(
            building=building,
            weather=weather,
            simulation_type=SimulationType.ECM,
            output_directory=output_directory / f"sample_{i:03d}",
            output_prefix=f"ecm_{i:03d}",
            ecm_parameters=ecm_parameters,
        )
        jobs.append(job)

    results = Parallel(n_jobs=n_samples, verbose=10, backend="loky")(
        delayed(_single_run)(job, config) for job in jobs
    )

    return results


def _single_run(job: SimulationJob, config: ConfigManager):
    IDF.setiddname(str(config.paths.idd_file))
    idf = IDF(str(job.building.idf_file_path))

    context = ECMContext(
        job=job,
        idf=idf,
    )

    executor = EnergyPlusExecutor()
    result_parser = ResultParser()
    file_cleaner = FileCleaner()

    service = ECMService(
        executor=executor,
        result_parser=result_parser,
        file_cleaner=file_cleaner,
        config=config,
    )
    result = service.run(context, context.job.ecm_parameters)  # type: ignore

    with open(context.job.output_directory / "result.pkl", "wb") as f:
        dump(result, f)

    with open(context.job.output_directory / "result.pkl", "rb") as f:
        test_result = load(f)

    return test_result


if __name__ == "__main__":
    test_batch_simulation()
