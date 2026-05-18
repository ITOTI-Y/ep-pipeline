from idfpy.sim import simulate as sim
from loguru import logger

from backend.models import SimulationJob, SimulationResult
from backend.services.interfaces import IEnergyPlusExecutor


class EnergyPlusExecutor(IEnergyPlusExecutor):
    def run(
        self,
        job: SimulationJob,
    ) -> SimulationResult:
        idf = job.idf
        output_prefix = job.output_prefix
        weather_file = job.weather.file_path
        output_directory = job.output_directory
        read_variables = job.read_variables
        job_id = job.id

        logger.info(f"Running EnergyPlus simulation: {output_prefix}")
        logger.debug(f"Weather file: {weather_file}")
        logger.debug(f"Output directory: {output_directory}")
        logger.debug(f"Output prefix: {output_prefix}")
        logger.debug(f"Read variables: {read_variables}")

        if idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")

        output_directory.mkdir(parents=True, exist_ok=True)
        idf.save(output_directory / f"{output_prefix}.idf")

        result = SimulationResult(
            job_id=job_id,
            building_type=job.building.building_type,
        )

        try:
            sim_result = sim(
                idf=idf,
                weather=weather_file,
                output_dir=output_directory,
                output_prefix=output_prefix,
                readvars=read_variables,
            )

            result.success = not (sim_result.err and sim_result.err.has_fatal)

            if result.success:
                logger.success(
                    f"EnergyPlus simulation completed successfully: {output_prefix}"
                )
            else:
                logger.error(
                    f"EnergyPlus simulation completed with errors: {result.errors}"
                )

        except Exception as e:
            logger.exception("Failed to run EnergyPlus: ")
            result.add_error(f"Failed to run EnergyPlus: {e}")

        return result
