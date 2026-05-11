import pvlib
from idfpy import IDF
from idfpy.models import (
    BuildingSurfaceDetailed,
    ElectricLoadCenterDistribution,
    ElectricLoadCenterGenerators,
    ElectricLoadCenterGeneratorsGeneratorOutputsItem,
    ElectricLoadCenterInverterPVWatts,
    ElectricLoadCenterInverterSimple,
    GeneratorPhotovoltaic,
    GeneratorPVWatts,
    PhotovoltaicPerformanceEquivalentOneDiode,
    PhotovoltaicPerformanceSandia,
    PhotovoltaicPerformanceSimple,
)
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from backend.models import SimulationJob, Surface
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class PVParameters(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
    )
    name: str = Field(..., description="The name of the PV system")
    cell_type: str = Field(..., description="The type of cell used in the PV system")
    number_of_cells_in_series: int = Field(
        ..., description="The number of cells in series"
    )
    active_area: float = Field(..., description="The active area of the PV system")
    transmittance_absorptance_product: float = Field(
        ..., description="The transmittance absorptance product"
    )
    shunt_resistance: float | None = Field(
        default=None, description="The shunt resistance"
    )
    short_circuit_current: float = Field(..., description="The short circuit current")
    open_circuit_voltage: float = Field(..., description="The open circuit voltage")
    reference_temperature: float = Field(..., description="The reference temperature")
    reference_insolation: float = Field(..., description="The reference insolation")
    module_current_at_maximum_power: float = Field(
        ..., description="The module current at maximum power"
    )
    module_voltage_at_maximum_power: float = Field(
        ..., description="The module voltage at maximum power"
    )
    temperature_coefficient_of_short_circuit_current: float = Field(
        ..., description="The temperature coefficient of short circuit current"
    )
    temperature_coefficient_of_open_circuit_voltage: float = Field(
        ..., description="The temperature coefficient of open circuit voltage"
    )
    nominal_operating_cell_temperature_test_cell_temperature: float = Field(
        ..., description="The nominal operating cell temperature test cell temperature"
    )


class PVApply(IApply):
    def __init__(self, config: ConfigManager, surfaces: list[Surface]):
        super().__init__()
        self._config = config
        self._surfaces = surfaces
        self._generators_and_surfaces: list[tuple[GeneratorPhotovoltaic, Surface]] = []

    def apply(self, job: SimulationJob) -> None:
        logger.info("Applying PV configuration")
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        pv_name, cec_module_data = self._get_pv_parameters()
        pv_performance = self._configure_pv_performance(
            job.idf, pv_name, cec_module_data
        )
        self._configure_pv_generator(job.idf, pv_performance)
        self._configure_loadcenter(job.idf, pv_performance)
        self._configure_inverter(job.idf)
        logger.info("PV configuration applied successfully")

    def _get_pv_parameters(
        self, pv_name: str = "Jinko_Solar_Co___Ltd_JKM200M_72"
    ) -> tuple[str, dict]:
        cec_modules = pvlib.pvsystem.retrieve_sam("cecmod")
        cec_modules = cec_modules.T
        cec_module_data = cec_modules.loc[pv_name]

        return pv_name, cec_module_data

    def _configure_pv_performance(
        self, idf: IDF, pv_name: str, cec_module_data: dict
    ) -> PhotovoltaicPerformanceEquivalentOneDiode:
        self._remove_objects(idf, PhotovoltaicPerformanceSimple)
        self._remove_objects(idf, PhotovoltaicPerformanceEquivalentOneDiode)
        self._remove_objects(idf, PhotovoltaicPerformanceSandia)

        pv_performance = PhotovoltaicPerformanceEquivalentOneDiode(
            name=pv_name + "_performance",
            cell_type="CrystallineSilicon",
            number_of_cells_in_series=cec_module_data["N_s"],
            active_area=cec_module_data["A_c"],
            transmittance_absorptance_product=0.95,
            shunt_resistance=None,
            short_circuit_current=cec_module_data["I_sc_ref"],
            open_circuit_voltage=cec_module_data["V_oc_ref"],
            reference_temperature=25.0,
            reference_insolation=1000.0,
            module_current_at_maximum_power=cec_module_data["I_mp_ref"],
            module_voltage_at_maximum_power=cec_module_data["V_mp_ref"],
            temperature_coefficient_of_short_circuit_current=cec_module_data[
                "alpha_sc"
            ],
            temperature_coefficient_of_open_circuit_voltage=cec_module_data["beta_oc"],
            nominal_operating_cell_temperature_test_cell_temperature=cec_module_data[
                "T_NOCT"
            ],
        )
        idf.add(pv_performance)

        logger.success("PV performance configured successfully")

        return pv_performance

    def _configure_pv_generator(
        self, idf: IDF, pv_performance: PhotovoltaicPerformanceEquivalentOneDiode
    ) -> None:
        self._remove_objects(idf, GeneratorPVWatts)
        self._remove_objects(idf, GeneratorPhotovoltaic)
        self._remove_objects(idf, ElectricLoadCenterGenerators)
        self._remove_objects(idf, ElectricLoadCenterDistribution)

        modified_count = 0
        for surface in self._surfaces:
            if (
                surface.type in ["Roof", "Wall"]
                and surface.sum_irradiation > self._config.pv.radiation_threshold
            ):
                surface_area = self._get_surface_area(idf, surface)
                if pv_performance.active_area is None:
                    logger.error("Active area is not set, skipping")
                    raise ValueError("Active area is not set")
                total_modules = max(
                    1,
                    int(
                        surface_area
                        / pv_performance.active_area
                        * self._config.pv.coverage[surface.type]
                    ),
                )

                pv_generator = GeneratorPhotovoltaic(
                    name=pv_performance.name + f"_generator_{modified_count}",
                    surface_name=surface.name,
                    photovoltaic_performance_object_type="PhotovoltaicPerformance:EquivalentOne-Diode",
                    module_performance_name=pv_performance.name,
                    heat_transfer_integration_mode="Decoupled",
                    number_of_series_strings_in_parallel=1,
                    number_of_modules_in_series=total_modules,
                )

                idf.add(pv_generator)

                self._generators_and_surfaces.append((pv_generator, surface))
                modified_count += 1

        logger.info(f"Added {modified_count} PV generator objects")
        logger.success("PV generator configured successfully")

    def _configure_loadcenter(
        self, idf: IDF, pv_performance: PhotovoltaicPerformanceEquivalentOneDiode
    ) -> None:
        self._remove_objects(idf, ElectricLoadCenterGenerators)
        self._remove_objects(idf, ElectricLoadCenterDistribution)

        if not self._generators_and_surfaces:
            logger.info(
                "No PV generators created; skipping ElectricLoadCenter configuration"
            )
            return

        def _get_power_output(surface: Surface) -> float:
            if (
                pv_performance.module_current_at_maximum_power is None
                or pv_performance.module_voltage_at_maximum_power is None
                or pv_performance.active_area is None
            ):
                logger.error(
                    "Module current at maximum power, module voltage at maximum power, or active area is not set, skipping"
                )
                raise ValueError(
                    "Module current at maximum power, module voltage at maximum power, or active area is not set"
                )
            return (
                pv_performance.module_current_at_maximum_power
                * pv_performance.module_voltage_at_maximum_power
                * self._get_surface_area(idf, surface)
                / pv_performance.active_area
                * self._config.pv.coverage[surface.type]
            )

        gen_list = ElectricLoadCenterGenerators(
            name="PV_Generator_List",
            generator_outputs=[
                ElectricLoadCenterGeneratorsGeneratorOutputsItem(
                    generator_name=generator.name,
                    generator_object_type="Generator:Photovoltaic",
                    generator_rated_electric_power_output=_get_power_output(surface),
                )
                for generator, surface in self._generators_and_surfaces
            ],
        )

        idf.add(gen_list)

        logger.info(
            f"Added {len(self._generators_and_surfaces)} generator objects to ElectricLoadCenter:Generators"
        )

        self._remove_objects(idf, ElectricLoadCenterDistribution)
        idf.add(
            ElectricLoadCenterDistribution(
                name="PV_Distribution",
                generator_list_name=gen_list.name,
                generator_operation_scheme_type="Baseload",
                generator_demand_limit_scheme_purchased_electric_demand_limit=None,
                generator_track_schedule_name_scheme_schedule_name="",
                generator_track_meter_scheme_meter_name="",
                electrical_buss_type="AlternatingCurrentWithStorage",
                inverter_name="PV_Inverter",
                electrical_storage_object_name="PV_Storage",
                storage_operation_scheme="TrackFacilityElectricDemandStoreExcessOnSite",
            )
        )

        logger.info("Added ElectricLoadCenter:Distribution object")
        logger.success("PV loadcenter configured successfully")

    def _configure_inverter(self, idf: IDF) -> None:
        self._remove_objects(idf, ElectricLoadCenterInverterPVWatts)
        self._remove_objects(idf, ElectricLoadCenterInverterSimple)

        idf.add(
            ElectricLoadCenterInverterSimple(
                name="PV_Inverter",
                radiative_fraction=0.0,
                inverter_efficiency=0.96,
            )
        )

        logger.success("PV inverter configured successfully")

    def _get_surface_area(self, idf: IDF, surface: Surface) -> float:
        surface_object = idf.all_of_type(BuildingSurfaceDetailed)
        for name, obj in surface_object.items():
            if name.upper() == surface.name.upper():
                return obj.area
        logger.error(f"Surface {surface.name} not found")
        raise ValueError(f"Surface {surface.name} not found")
