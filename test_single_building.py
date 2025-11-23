from pathlib import Path

from eppy.modeleditor import IDF

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.models import (
    Building,
    BuildingType,
    SimulationJob,
    Weather,
)
from backend.models.enums import SimulationType
from backend.services.simulation import BaselineService, FileCleaner, ResultParser
from backend.utils.config import ConfigManager


def run_single_building_simulation():
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

    IDF.setiddname(str(config.paths.idd_file))
    job = SimulationJob(
        building=building,
        weather=weather,
        simulation_type=SimulationType.BASELINE,
        output_directory=config.paths.baseline_dir / building.name,
        output_prefix="baseline_",
        read_variables=True,
    )

    executor = EnergyPlusExecutor()
    result_parser = ResultParser()
    file_cleaner = FileCleaner()

    service = BaselineService(
        executor=executor,
        result_parser=result_parser,
        file_cleaner=file_cleaner,
        config=config,
        job=job,
    )

    job.idf = IDF(str(building.idf_file_path))
    result = service.run()

    print(result)


if __name__ == "__main__":
    run_single_building_simulation()
