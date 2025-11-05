from pathlib import Path

from eppy.modeleditor import IDF

from backend.bases.energyplus.executor import EnergyPlusExecutor
from backend.domain.models import (
    BaselineContext,
    Building,
    BuildingType,
    SimulationJob,
    Weather,
)
from backend.domain.models.enums import SimulationType
from backend.services.simulation import BaselineService, FileCleaner, ResultParser
from backend.utils.config import ConfigManager


def run_single_building_simulation():
    config = ConfigManager(Path("backend/configs"))

    idf_file_path = config.paths.idf_files[1]

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

    job = SimulationJob(
        building=building,
        weather=weather,
        simulation_type=SimulationType.BASELINE,
        output_directory=config.paths.baseline_dir / building.name,
        output_prefix="baseline_",
        read_variables=True,
    )

    IDF.setiddname(str(config.paths.idd_file))
    idf = IDF(str(building.idf_file_path))

    context = BaselineContext(
        job=job,
        idf=idf,
    )

    executor = EnergyPlusExecutor()
    result_parser = ResultParser()
    file_cleaner = FileCleaner()

    service = BaselineService(
        executor=executor,
        result_parser=result_parser,
        file_cleaner=file_cleaner,
        config=config,
    )
    result = service.run(context)


if __name__ == "__main__":
    run_single_building_simulation()
