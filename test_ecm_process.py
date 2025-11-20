from pathlib import Path
from pickle import dump, load

from eppy.modeleditor import IDF

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


def test_ecm_process():
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

    sampler = ParameterSampler(config=config)
    ecm_samples = sampler.sample(n_samples=10, building_type=building_type)

    IDF.setiddname(str(config.paths.idd_file))
    job = SimulationJob(
        building=building,
        weather=weather,
        simulation_type=SimulationType.ECM,
        output_directory=config.paths.ecm_dir / building.name,
        output_prefix="ecm_",
        read_variables=True,
        idf=IDF(str(building.idf_file_path)),
        ecm_parameters=ecm_samples[0],
    )

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
    result = service.run()

    with open(config.paths.ecm_dir / building.name / "result.pkl", "wb") as f:
        dump(result, f)

    with open(config.paths.ecm_dir / building.name / "result.pkl", "rb") as f:
        test_result = load(f)
    print(test_result)


if __name__ == "__main__":
    test_ecm_process()
