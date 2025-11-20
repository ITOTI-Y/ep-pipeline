from pathlib import Path
from pickle import dump, load

from eppy.modeleditor import IDF
from joblib import Parallel, delayed

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.models import (
    Building,
    BuildingType,
    SimulationJob,
    Weather,
)
from backend.models.enums import SimulationType
from backend.services.optimization import ParameterSampler
from backend.services.simulation import ECMService, FileCleaner, ResultParser
from backend.utils.config import ConfigManager


def test_batch_simulation():
    config = ConfigManager(Path("backend/configs"))

    idf_file_path = config.paths.idf_files[2]
    building_type = BuildingType.from_str(idf_file_path.stem)

    building = Building(
        name=idf_file_path.stem,
        building_type=building_type,
        location="Chicago",
        idf_file_path=idf_file_path,
    )

    weather = Weather(
        file_path=config.paths.ftmy_files[0],
        location="Chicago",
    )

    output_directory = config.paths.ecm_dir / building.name

    n_samples = 2

    sampler = ParameterSampler(config=config)
    ecm_samples = sampler.sample(
        n_samples=n_samples, building_type=building_type
    )

    jobs = []
    for i, ecm_parameters in enumerate(ecm_samples):
        IDF.setiddname(str(config.paths.idd_file))
        job = SimulationJob(
            building=building,
            weather=weather,
            simulation_type=SimulationType.ECM,
            output_directory=output_directory / f"sample_{i:03d}",
            output_prefix=f"ecm_{i:03d}",
            ecm_parameters=ecm_parameters,
            idf=IDF(str(building.idf_file_path)),
        )
        jobs.append(job)

    results = Parallel(n_jobs=n_samples, verbose=10, backend="loky")(
        delayed(_single_run)(job, config) for job in jobs
    )

    return results


def _single_run(job: SimulationJob, config: ConfigManager):
    IDF.setiddname(str(config.paths.idd_file))
    job.idf = IDF(str(job.building.idf_file_path))
    executor = EnergyPlusExecutor()
    result_parser = ResultParser()
    file_cleaner = FileCleaner()

    service = ECMService(
        executor=executor,
        result_parser=result_parser,
        file_cleaner=file_cleaner,
        config=config,
        job=job,
    )

    if job.ecm_parameters is None:
        raise ValueError("ECM parameters are not set")
    result = service.run()

    with open(job.output_directory / "result.pkl", "wb") as f:
        dump(result, f)

    with open(job.output_directory / "result.pkl", "rb") as f:
        test_result = load(f)

    return test_result


if __name__ == "__main__":
    test_batch_simulation()
