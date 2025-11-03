import re
from pathlib import Path

from eppy.modeleditor import IDF
from loguru import logger

from backend.services.interfaces import ExecutionResult, IEnergyPlusExecutor


class EnergyPlusExecutor(IEnergyPlusExecutor):
    _SEVERE_PATTERN = re.compile(r"\*\*\s+Severe\s+\*\*.*", re.IGNORECASE)
    _FATAL_PATTERN = re.compile(r"\*\*\s+Fatal\s+\*\*.*", re.IGNORECASE)
    _WARNING_PATTERN = re.compile(r"\*\*\s+Warning\s+\*\*.*", re.IGNORECASE)

    def __init__(
        self,
        idd_path: Path | None = None,
        energyplus_path: Path | None = None,
    ):
        self._idd_path = idd_path
        self._energyplus_path = energyplus_path
        self._logger = logger.bind(module=self.__class__.__name__)

        if idd_path:
            IDF.setiddname(str(idd_path))

    def run(
        self,
        idf: IDF,
        weather_file: Path,
        output_directory: Path,
        output_prefix: str,
        read_variables: bool = True,
    ) -> ExecutionResult:
        self._logger.info(f"Running EnergyPlus simulation: {output_prefix}")
        self._logger.debug(f"Weather file: {weather_file}")
        self._logger.debug(f"Output directory: {output_directory}")
        self._logger.debug(f"Output prefix: {output_prefix}")
        self._logger.debug(f"Read variables: {read_variables}")

        result = ExecutionResult(
            success=True,
            return_code=0,
            stdout="",
            stderr="",
            output_directory=output_directory,
        )

        try:
            idf.epw = str(weather_file)
            idf.run(
                weather=str(weather_file),
                output_directory=str(output_directory),
                output_prefix=output_prefix,
                readvars=read_variables,
                verbose="v",
            )

            err_file = output_directory / f"{output_prefix}out.err"
            if err_file.exists():
                self._parse_error_file(err_file, result)

            if result.success:
                self._logger.success(
                    f"EnergyPlus simulation completed successfully: {output_prefix}"
                )
            else:
                self._logger.error(
                    f"EnergyPlus simulation completed with errors: {output_prefix}"
                )
                for error in result.errors:
                    self._logger.error(f"  - {error}")
            return result

        except Exception as e:
            self._logger.exception("Failed to run EnergyPlus: ")

            result = ExecutionResult(
                success=False,
                return_code=1,
                stdout="",
                stderr=str(e),
                output_directory=output_directory,
            )
            result.add_error(f"Failed to run EnergyPlus: {e}")
            return result

    def _parse_error_file(
        self,
        err_file: Path,
        result: ExecutionResult,
    ) -> None:
        try:
            content = err_file.read_text(encoding="utf-8", errors="ignore")

            severe_errors = self._SEVERE_PATTERN.findall(content)
            fatal_errors = self._FATAL_PATTERN.findall(content)
            warnings = self._WARNING_PATTERN.findall(content)

            for error in severe_errors:
                result.add_error(error.strip())
            for error in fatal_errors:
                result.add_error(error.strip())
            for warning in warnings:
                result.add_warning(warning.strip())

        except FileNotFoundError:
            self._logger.warning(f"Error file not found: {err_file}")
        except Exception:
            self._logger.exception("Failed to parse error file: ")
