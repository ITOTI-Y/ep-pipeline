import pvlib
from eppy.geometry.surface import area
from eppy.modeleditor import IDF
from loguru import logger

from backend.models import SimulationJob, Surface
from backend.services.configuration.iapply import IApply
from backend.utils.config import ConfigManager


class PVApply(IApply):
    def __init__(self, config: ConfigManager, surfaces: list[Surface]):
        super().__init__()
        self._config = config
        self._surfaces = surfaces
        self._generators_and_surfaces = []

    def apply(self, job: SimulationJob) -> None:
        logger.info("Applying PV configuration")
        if job.idf is None:
            logger.error("IDF is not set, skipping")
            raise ValueError("IDF is not set")
        pv_parameters = self._get_pv_parameters()
        self._configure_pv_performance(job.idf, pv_parameters)
        self._configure_pv_generator(job.idf, pv_parameters)
        self._configure_loadcenter(job.idf, pv_parameters)
        self._configure_inverter(job.idf)
        logger.info("PV configuration applied successfully")

    def _get_pv_parameters(
        self, pv_name: str = "Jinko_Solar_Co___Ltd_JKM200M_72"
    ) -> dict:
        cec_modules = pvlib.pvsystem.retrieve_sam("cecmod")
        cec_modules = cec_modules.T
        cec_module_data = cec_modules.loc[pv_name]

        pv_parameters = {
            "Name": pv_name,
            "Cell_type": "CrystallineSilicon",
            "Number_of_Cells_in_Series": cec_module_data["N_s"],
            "Active_Area": cec_module_data["A_c"],
            "Transmittance_Absorptance_Product": 0.95,
            "Shunt_Resistance": "",
            "Short_Circuit_Current": cec_module_data["I_sc_ref"],
            "Open_Circuit_Voltage": cec_module_data["V_oc_ref"],
            "Reference_Temperature": 25.0,
            "Reference_Insolation": 1000.0,
            "Module_Current_at_Maximum_Power": cec_module_data["I_mp_ref"],
            "Module_Voltage_at_Maximum_Power": cec_module_data["V_mp_ref"],
            "Temperature_Coefficient_of_Short_Circuit_Current": cec_module_data[
                "alpha_sc"
            ],
            "Temperature_Coefficient_of_Open_Circuit_Voltage": cec_module_data[
                "beta_oc"
            ],
            "Nominal_Operating_Cell_Temperature_Test_Cell_Temperature": cec_module_data[
                "T_NOCT"
            ],
        }

        return pv_parameters

    def _configure_pv_performance(self, idf: IDF, pv_parameters: dict) -> None:
        self._remove_objects(idf, "PhotovoltaicPerformance:Simple")
        self._remove_objects(idf, "PhotovoltaicPerformance:EquivalentOne-Diode")
        self._remove_objects(idf, "PhotovoltaicPerformance:Sandia")

        pv_performance = idf.newidfobject(
            "PhotovoltaicPerformance:EquivalentOne-Diode", **pv_parameters
        )
        pv_performance.Name = pv_parameters["Name"] + "_performance"

        logger.success("PV performance configured successfully")

    def _configure_pv_generator(
        self, idf: IDF, pv_parameters: dict
    ) -> None:
        self._remove_objects(idf, "Generator:PVWatts")
        self._remove_objects(idf, "Generator:Photovoltaic")
        self._remove_objects(idf, "ElectricLoadCenter:Generators")
        self._remove_objects(idf, "ElectricLoadCenter:Distribution")

        modified_count = 0
        for surface in self._surfaces:
            if (
                surface.type in ["Roof", "Wall"]
                and surface.sum_irradiation > self._config.pv.radiation_threshold
            ):
                surface_area = self._get_surace_area(idf, surface)
                pv_generator = idf.newidfobject("Generator:Photovoltaic")
                pv_generator.Name = pv_parameters["Name"] + f"_generator_{modified_count}"
                surface_object = idf.getobject('BuildingSurface:Detailed', surface.name)
                if surface_object is None:
                    logger.error(f"Surface {surface.name} not found")
                    raise ValueError(f"Surface {surface.name} not found")
                pv_generator.Surface_Name = surface_object.Name
                pv_generator.Photovoltaic_Performance_Object_Type = (
                    "PhotovoltaicPerformance:EquivalentOne-Diode"
                )
                pv_generator.Module_Performance_Name = (
                    pv_parameters["Name"] + "_performance"
                )
                pv_generator.Heat_Transfer_Integration_Mode = "Decoupled"
                total_modules = max(1, int(surface_area / pv_parameters["Active_Area"] * 0.8))
                pv_generator.Number_of_Series_Strings_in_Parallel = 1
                pv_generator.Number_of_Modules_in_Series = total_modules

                self._generators_and_surfaces.append((pv_generator, surface))
                modified_count += 1

        logger.info(f"Added {modified_count} PV generator objects")
        logger.success("PV generator configured successfully")

    def _configure_loadcenter(self, idf: IDF, pv_parameters: dict) -> None:
        self._remove_objects(idf, "ElectricLoadCenter:Generators")
        self._remove_objects(idf, "ElectricLoadCenter:Distribution")

        if not self._generators_and_surfaces:
            logger.info("No PV generators created; skipping ElectricLoadCenter configuration")
            return

        gen_list = idf.newidfobject("ElectricLoadCenter:Generators")
        gen_list.Name = "PV_Generator_List"

        for index, (generator, surface) in enumerate(self._generators_and_surfaces):

            surface_area = self._get_surace_area(idf, surface)
            power_output = pv_parameters["Module_Current_at_Maximum_Power"] * pv_parameters["Module_Voltage_at_Maximum_Power"] * surface_area / pv_parameters["Active_Area"]
            setattr(gen_list, f"Generator_{index + 1}_Name", generator.Name)
            setattr(gen_list, f"Generator_{index + 1}_Object_Type", "Generator:Photovoltaic")
            setattr(gen_list, f"Generator_{index + 1}_Rated_Electric_Power_Output", power_output)

        logger.info(f"Added {index + 1} generator objects to ElectricLoadCenter:Generators")

        self._remove_objects(idf, "ElectricLoadCenter:Distribution")
        distribution = idf.newidfobject("ElectricLoadCenter:Distribution")
        distribution.Name = "PV_Distribution"
        distribution.Generator_List_Name = gen_list.Name
        distribution.Generator_Operation_Scheme_Type = "Baseload"
        distribution.Generator_Demand_Limit_Scheme_Purchased_Electric_Demand_Limit = ''
        distribution.Generator_Track_Schedule_Name_Scheme_Schedule_Name = ''
        distribution.Generator_Track_Meter_Scheme_Meter_Name = ''
        distribution.Electrical_Buss_Type = 'AlternatingCurrentWithStorage'
        distribution.Inverter_Name = "PV_Inverter"
        distribution.Electrical_Storage_Object_Name = 'PV_Storage'
        distribution.Storage_Operation_Scheme = 'TrackFacilityElectricDemandStoreExcessOnSite'

        logger.info("Added ElectricLoadCenter:Distribution object")
        logger.success("PV loadcenter configured successfully")

    def _configure_inverter(self, idf: IDF) -> None:
        self._remove_objects(idf, "ElectricLoadCenter:Inverter:PVWatts")
        self._remove_objects(idf, "ElectricLoadCenter:Inverter:Simple")

        inverter = idf.newidfobject("ElectricLoadCenter:Inverter:Simple")
        inverter.Name = "PV_Inverter"
        inverter.Radiative_Fraction = 0.0
        inverter.Inverter_Efficiency = 0.96
        logger.success("PV inverter configured successfully")

    def _get_surace_area(self, idf: IDF, surface: Surface) -> float:
        surface_object = idf.getobject("BUILDINGSURFACE:DETAILED", surface.name)

        if surface_object is None:
            logger.error(f"Surface {surface.name} not found")
            raise ValueError(f"Surface {surface.name} not found")

        num_vertices = int(surface_object.Number_of_Vertices)
        vertices = []
        for i in range(1, num_vertices + 1):
            x = float(getattr(surface_object, f"Vertex_{i}_Xcoordinate"))
            y = float(getattr(surface_object, f"Vertex_{i}_Ycoordinate"))
            z = float(getattr(surface_object, f"Vertex_{i}_Zcoordinate"))
            vertices.append((x, y, z))

        return area(vertices)
