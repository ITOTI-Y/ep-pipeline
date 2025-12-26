#!/usr/bin/env python3
"""
PV Simulation Results Visualization Report Generator

Generates academic paper-quality figures for Building and Environment journal.
Compares PV results with baseline results across all building-climate combinations.

Usage:
    uv run python -m backend.visualization.main

Output:
    backend/output/visualization/figures/*.pdf
"""

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import pandas as pd
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.visualization.config import (
    ALL_BUILDINGS,
    ALL_CLIMATES,
    BUILDINGS_WITH_STORAGE,
    GRID_EMISSION_FACTOR,
    FigureConfig,
)
from backend.visualization.data_loader import DataLoader
from backend.visualization.data_processor import DataProcessor
from backend.visualization.figures import (
    CarbonReductionFigure,
    ClimateComparisonFigure,
    ClimateEUITrendFigure,
    ECMCorrelationMatrix,
    ECMSensitivityHeatmap,
    EnergySankeyFigure,
    EUIWaterfallFigure,
    LoadClusteringFigure,
    LoadPVMatchFigure,
    MonthlyCurtailmentFigure,
    MonthlyPVGenerationFigure,
    PeakReductionFigure,
    PerformanceRadarFigure,
    PVEfficiencyTempFigure,
    SelfConsumptionFigure,
    SensitivityIndexFigure,
    StorageSOCFigure,
    TypicalDayStorageFigure,
)
from backend.visualization.metrics import MetricsCalculator


class VisualizationReport:
    """
    Main visualization report generator.

    Generates all figures for PV simulation analysis.
    """

    def __init__(
        self,
        base_path: Path | str | None = None,
        output_dir: Path | str | None = None,
    ):
        """
        Initialize report generator.

        Args:
            base_path: Base path to simulation output
            output_dir: Output directory for figures
        """
        self.config = FigureConfig()
        self.loader = DataLoader(base_path)
        self.processor = DataProcessor()
        self.metrics = MetricsCalculator()

        if output_dir is None:
            self.output_dir = self.loader.base_path / "visualization" / "figures"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded data
        self._pv_data_cache: dict[str, dict[str, pd.DataFrame]] = {}
        self._result_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._metrics_cache: dict[str, dict[str, dict]] = {}

    def generate_all(self) -> None:
        """Generate all visualization figures."""
        logger.info("Starting visualization report generation")
        logger.info(f"Output directory: {self.output_dir}")

        # Check data availability
        availability = self.loader.check_data_availability()
        logger.info(f"Data availability: {availability}")

        # Load all data
        self._load_all_data()

        # Generate figures by category
        self._generate_storage_figures()
        self._generate_pv_figures()
        self._generate_ecm_figures()
        self._generate_energy_figures()
        self._generate_supplementary_figures()

        logger.info("Visualization report generation complete!")

    def _load_all_data(self) -> None:
        """Load and cache all required data."""
        logger.info("Loading simulation data...")

        # Load PV hourly data
        for building in ALL_BUILDINGS:
            self._pv_data_cache[building] = {}
            for weather in ALL_CLIMATES:
                try:
                    raw_df = self.loader.load_pv_hourly(building, weather)
                    processed_df = self.processor.process_pv_data(raw_df, building)
                    self._pv_data_cache[building][weather] = processed_df
                    logger.debug(f"Loaded PV data: {building}/{weather}")
                except FileNotFoundError:
                    logger.warning(f"PV data not found: {building}/{weather}")

        # Load result pickles
        for result_type in ["baseline", "optimization", "pv"]:
            self._result_cache[result_type] = {}
            for building in ALL_BUILDINGS:
                self._result_cache[result_type][building] = {}
                for weather in ALL_CLIMATES:
                    try:
                        result = self.loader.load_result_pkl(result_type, building, weather)
                        self._result_cache[result_type][building][weather] = result
                    except FileNotFoundError:
                        pass

        # Calculate metrics for all combinations
        for building in ALL_BUILDINGS:
            self._metrics_cache[building] = {}
            for weather in ALL_CLIMATES:
                if building in self._pv_data_cache and weather in self._pv_data_cache[building]:
                    df = self._pv_data_cache[building][weather]
                    baseline_df = None
                    # Try to load baseline hourly data
                    try:
                        baseline_raw = self.loader.load_baseline_hourly(building, weather)
                        baseline_df = self.processor.process_pv_data(baseline_raw, building)
                    except FileNotFoundError:
                        pass

                    metrics = self.metrics.calc_all_metrics(df, building, baseline_df)
                    self._metrics_cache[building][weather] = metrics

        logger.info("Data loading complete")

    def _generate_storage_figures(self) -> None:
        """Generate storage-related figures (F1-F3)."""
        logger.info("Generating storage figures (F1-F3)...")

        for building in BUILDINGS_WITH_STORAGE:
            for weather in ALL_CLIMATES:
                if building not in self._pv_data_cache:
                    continue
                if weather not in self._pv_data_cache[building]:
                    continue

                df = self._pv_data_cache[building][weather]
                metrics = self._metrics_cache.get(building, {}).get(weather, {})
                curtailment = metrics.get("curtailment", {}).get("hourly_curtailment")

                # F1: Storage SOC curve
                try:
                    fig_soc = StorageSOCFigure(self.config)
                    fig = fig_soc.plot(df, building, weather, curtailment)
                    fig_soc.save(fig, f"F01_storage_soc_{building}_{weather}", self.output_dir)
                    fig_soc.close(fig)
                    logger.debug(f"Generated F1 for {building}/{weather}")
                except Exception as e:
                    logger.error(f"Failed F1 for {building}/{weather}: {e}")

                # F2: Typical day storage operation
                try:
                    summer_df, winter_df = self.processor.get_summer_winter_days(df)
                    if len(summer_df) > 0 and len(winter_df) > 0:
                        fig_typical = TypicalDayStorageFigure(self.config)
                        fig = fig_typical.plot(summer_df, winter_df, building)
                        fig_typical.save(fig, f"F02_typical_day_{building}_{weather}", self.output_dir)
                        fig_typical.close(fig)
                        logger.debug(f"Generated F2 for {building}/{weather}")
                except Exception as e:
                    logger.error(f"Failed F2 for {building}/{weather}: {e}")

                # F3: Monthly curtailment
                try:
                    monthly_curtailment = metrics.get("curtailment", {}).get("monthly_curtailment")
                    if monthly_curtailment is not None:
                        monthly_pv = df.groupby("month")["pv_generation_kw"].sum()
                        fig_curtail = MonthlyCurtailmentFigure(self.config)
                        fig = fig_curtail.plot(monthly_curtailment, monthly_pv, building, weather)
                        fig_curtail.save(fig, f"F03_curtailment_{building}_{weather}", self.output_dir)
                        fig_curtail.close(fig)
                        logger.debug(f"Generated F3 for {building}/{weather}")
                except Exception as e:
                    logger.error(f"Failed F3 for {building}/{weather}: {e}")

    def _generate_pv_figures(self) -> None:
        """Generate PV-related figures (F4-F6)."""
        logger.info("Generating PV figures (F4-F6)...")

        # F4: Monthly PV generation for each combination
        for building in ALL_BUILDINGS:
            for weather in ALL_CLIMATES:
                if building not in self._pv_data_cache:
                    continue
                if weather not in self._pv_data_cache[building]:
                    continue

                df = self._pv_data_cache[building][weather]

                try:
                    fig_monthly = MonthlyPVGenerationFigure(self.config)
                    fig = fig_monthly.plot(df, building, weather)
                    fig_monthly.save(fig, f"F04_monthly_pv_{building}_{weather}", self.output_dir)
                    fig_monthly.close(fig)
                    logger.debug(f"Generated F4 for {building}/{weather}")
                except Exception as e:
                    logger.error(f"Failed F4 for {building}/{weather}: {e}")

        # F5: Climate comparison for each building
        for building in ALL_BUILDINGS:
            if building not in self._pv_data_cache:
                continue

            annual_pv = {}
            for weather in ALL_CLIMATES:
                if weather in self._pv_data_cache[building]:
                    df = self._pv_data_cache[building][weather]
                    if "pv_generation_kw" in df.columns:
                        annual_pv[weather] = df["pv_generation_kw"].sum()

            if annual_pv:
                try:
                    fig_climate = ClimateComparisonFigure(self.config)
                    fig = fig_climate.plot(annual_pv, building)
                    fig_climate.save(fig, f"F05_climate_comparison_{building}", self.output_dir)
                    fig_climate.close(fig)
                    logger.debug(f"Generated F5 for {building}")
                except Exception as e:
                    logger.error(f"Failed F5 for {building}: {e}")

        # F6: Load-PV matching (sample week in summer)
        for building in ALL_BUILDINGS:
            weather = "TMY"  # Use TMY as representative
            if building not in self._pv_data_cache:
                continue
            if weather not in self._pv_data_cache[building]:
                continue

            df = self._pv_data_cache[building][weather]

            try:
                fig_match = LoadPVMatchFigure(self.config)
                # Summer week (around July)
                fig = fig_match.plot(df, building, weather, day_range=(182, 188))
                fig_match.save(fig, f"F06_load_pv_match_{building}_{weather}", self.output_dir)
                fig_match.close(fig)
                logger.debug(f"Generated F6 for {building}")
            except Exception as e:
                logger.error(f"Failed F6 for {building}: {e}")

    def _generate_ecm_figures(self) -> None:
        """Generate ECM sensitivity figures (F7-F8)."""
        logger.info("Generating ECM figures (F7-F8)...")

        try:
            ecm_df = self.loader.load_ecm_results()
        except FileNotFoundError:
            logger.warning("ECM results not found, skipping ECM figures")
            return

        # F7: Sensitivity heatmap (overall)
        try:
            fig_heatmap = ECMSensitivityHeatmap(self.config)
            fig = fig_heatmap.plot(ecm_df)
            fig_heatmap.save(fig, "F07_ecm_sensitivity_heatmap", self.output_dir)
            fig_heatmap.close(fig)
            logger.info("Generated F7: ECM sensitivity heatmap")
        except Exception as e:
            logger.error(f"Failed F7: {e}")

        # F7b: Sensitivity heatmap by building
        try:
            fig_heatmap_building = ECMSensitivityHeatmap(self.config)
            fig = fig_heatmap_building.plot_by_building(ecm_df, target="total_site_eui")
            fig_heatmap_building.save(fig, "F07b_ecm_sensitivity_by_building", self.output_dir)
            fig_heatmap_building.close(fig)
            logger.info("Generated F7b: ECM sensitivity by building")
        except Exception as e:
            logger.error(f"Failed F7b: {e}")

        # F8: Sensitivity index for each building
        if "building_type" in ecm_df.columns:
            buildings = ecm_df["building_type"].unique()
            for building in buildings:
                try:
                    fig_index = SensitivityIndexFigure(self.config)
                    fig = fig_index.plot(ecm_df, building)
                    fig_index.save(fig, f"F08_sensitivity_index_{building}", self.output_dir)
                    fig_index.close(fig)
                    logger.debug(f"Generated F8 for {building}")
                except Exception as e:
                    logger.error(f"Failed F8 for {building}: {e}")

        # F8b: Comparison across buildings
        try:
            fig_comparison = SensitivityIndexFigure(self.config)
            fig = fig_comparison.plot_comparison(ecm_df)
            fig_comparison.save(fig, "F08b_sensitivity_comparison", self.output_dir)
            fig_comparison.close(fig)
            logger.info("Generated F8b: Sensitivity comparison")
        except Exception as e:
            logger.error(f"Failed F8b: {e}")

    def _generate_energy_figures(self) -> None:
        """Generate energy analysis figures (F9-F12)."""
        logger.info("Generating energy figures (F9-F12)...")

        # Prepare EUI data - collect both total_site_eui and net_site_eui
        eui_by_building: dict[str, dict[str, dict[str, float]]] = {}
        net_eui_by_building: dict[str, dict[str, dict[str, float]]] = {}

        for building in ALL_BUILDINGS:
            eui_by_building[building] = {}
            net_eui_by_building[building] = {}
            for weather in ALL_CLIMATES:
                eui_data = {}
                net_eui_data = {}
                for stage in ["baseline", "optimization", "pv"]:
                    if stage in self._result_cache:
                        if building in self._result_cache[stage]:
                            if weather in self._result_cache[stage][building]:
                                result = self._result_cache[stage][building][weather]
                                eui_data[stage] = getattr(result, "total_site_eui", 0) or 0
                                net_eui_data[stage] = getattr(result, "net_site_eui", 0) or 0

                if eui_data:
                    eui_by_building[building][weather] = eui_data
                if net_eui_data:
                    net_eui_by_building[building][weather] = net_eui_data

        # F9: EUI waterfall for each building (TMY)
        for building in ALL_BUILDINGS:
            weather = "TMY"
            if building not in eui_by_building:
                logger.warning(f"F09: No EUI data for {building}")
                continue
            if weather not in eui_by_building[building]:
                logger.warning(f"F09: No {weather} data for {building}")
                continue

            eui_data = eui_by_building[building][weather]
            logger.debug(f"F09 {building}/{weather}: {eui_data}")

            # Allow partial data - use 0 for missing stages
            baseline = eui_data.get("baseline", 0)
            optimization = eui_data.get("optimization", baseline)  # Default to baseline if missing
            pv = eui_data.get("pv", optimization)  # Default to optimization if missing

            if baseline > 0:  # Only plot if we have at least baseline
                try:
                    fig_waterfall = EUIWaterfallFigure(self.config)
                    fig = fig_waterfall.plot(
                        baseline,
                        optimization,
                        pv,
                        building,
                        weather,
                    )
                    fig_waterfall.save(fig, f"F09_eui_waterfall_{building}_{weather}", self.output_dir)
                    fig_waterfall.close(fig)
                    logger.debug(f"Generated F9 for {building}")
                except Exception as e:
                    logger.error(f"Failed F9 for {building}: {e}")

        # F10: Climate EUI trend for each building - USE NET_SITE_EUI
        for building in ALL_BUILDINGS:
            if building not in net_eui_by_building:
                continue

            try:
                fig_trend = ClimateEUITrendFigure(self.config)
                fig = fig_trend.plot(net_eui_by_building[building], building)
                fig_trend.save(fig, f"F10_climate_eui_trend_{building}", self.output_dir)
                fig_trend.close(fig)
                logger.debug(f"Generated F10 for {building}")
            except Exception as e:
                logger.error(f"Failed F10 for {building}: {e}")

        # F11: Self-consumption analysis
        try:
            scr_ssr_data: dict[str, dict[str, dict]] = {}
            for building in ALL_BUILDINGS:
                scr_ssr_data[building] = {}
                for weather in ALL_CLIMATES:
                    metrics = self._metrics_cache.get(building, {}).get(weather, {})
                    sc = metrics.get("self_consumption", {})
                    if sc:
                        scr_ssr_data[building][weather] = sc

            if scr_ssr_data:
                fig_scr = SelfConsumptionFigure(self.config)
                fig = fig_scr.plot(scr_ssr_data)
                fig_scr.save(fig, "F11_self_consumption_analysis", self.output_dir)
                fig_scr.close(fig)
                logger.info("Generated F11: Self-consumption analysis")
        except Exception as e:
            logger.error(f"Failed F11: {e}")

        # F12: Peak reduction analysis
        try:
            peak_data: dict[str, dict[str, dict]] = {}
            for building in ALL_BUILDINGS:
                peak_data[building] = {}
                for weather in ALL_CLIMATES:
                    metrics = self._metrics_cache.get(building, {}).get(weather, {})
                    pr = metrics.get("peak_reduction", {})
                    if pr:
                        peak_data[building][weather] = pr

            if peak_data:
                fig_peak = PeakReductionFigure(self.config)
                fig = fig_peak.plot(peak_data)
                fig_peak.save(fig, "F12_peak_reduction", self.output_dir)
                fig_peak.close(fig)
                logger.info("Generated F12: Peak reduction analysis")
        except Exception as e:
            logger.error(f"Failed F12: {e}")

    def _generate_supplementary_figures(self) -> None:
        """Generate supplementary research figures (S1-S8)."""
        logger.info("Generating supplementary figures (S1-S8)...")

        # S2: ECM correlation matrix
        try:
            ecm_df = self.loader.load_ecm_results()
            fig_corr = ECMCorrelationMatrix(self.config)
            fig = fig_corr.plot(ecm_df)
            fig_corr.save(fig, "S02_ecm_correlation_matrix", self.output_dir)
            fig_corr.close(fig)
            logger.info("Generated S2: ECM correlation matrix")
        except Exception as e:
            logger.error(f"Failed S2: {e}")

        # S4: PV efficiency vs temperature
        for building in ALL_BUILDINGS:
            weather = "TMY"
            if building not in self._pv_data_cache:
                continue
            if weather not in self._pv_data_cache[building]:
                continue

            df = self._pv_data_cache[building][weather]

            try:
                fig_eff = PVEfficiencyTempFigure(self.config)
                fig = fig_eff.plot(df, building, weather)
                fig_eff.save(fig, f"S04_pv_efficiency_{building}", self.output_dir)
                fig_eff.close(fig)
                logger.debug(f"Generated S4 for {building}")
            except Exception as e:
                logger.error(f"Failed S4 for {building}: {e}")

        # S5: Energy flow diagram (representative)
        for building in ALL_BUILDINGS[:2]:  # Just first two buildings
            weather = "TMY"
            metrics = self._metrics_cache.get(building, {}).get(weather, {})
            sc = metrics.get("self_consumption", {})

            if sc:
                energy_flows = {
                    "pv_generation": sc.get("total_pv_kwh", 0),
                    "self_consumed": sc.get("self_consumed_kwh", 0),
                    "grid_export": max(0, sc.get("total_pv_kwh", 0) - sc.get("self_consumed_kwh", 0)),
                    "grid_import": max(0, sc.get("total_demand_kwh", 0) - sc.get("self_consumed_kwh", 0)),
                    "demand": sc.get("total_demand_kwh", 0),
                }

                try:
                    fig_sankey = EnergySankeyFigure(self.config)
                    fig = fig_sankey.plot(energy_flows, building, weather)
                    fig_sankey.save(fig, f"S05_energy_flow_{building}", self.output_dir)
                    fig_sankey.close(fig)
                    logger.debug(f"Generated S5 for {building}")
                except Exception as e:
                    logger.error(f"Failed S5 for {building}: {e}")

        # S6: Load clustering
        for building in ALL_BUILDINGS[:2]:
            weather = "TMY"
            if building not in self._pv_data_cache:
                continue
            if weather not in self._pv_data_cache[building]:
                continue

            df = self._pv_data_cache[building][weather]

            try:
                fig_cluster = LoadClusteringFigure(self.config)
                fig = fig_cluster.plot(df, building)
                fig_cluster.save(fig, f"S06_load_clustering_{building}", self.output_dir)
                fig_cluster.close(fig)
                logger.debug(f"Generated S6 for {building}")
            except Exception as e:
                logger.error(f"Failed S6 for {building}: {e}")

        # S7: Carbon reduction
        try:
            carbon_data: dict[str, dict[str, float]] = {}
            for building in ALL_BUILDINGS:
                weather = "TMY"
                metrics = self._metrics_cache.get(building, {}).get(weather, {})
                carbon = metrics.get("carbon", {})

                # Include all buildings, even if carbon data is incomplete
                baseline_carbon = carbon.get("baseline_carbon_kg", 0) if carbon else 0
                pv_carbon = carbon.get("pv_system_carbon_kg", 0) if carbon else 0

                # If no carbon data from metrics, try to calculate from PV data
                if baseline_carbon == 0 and pv_carbon == 0:
                    if building in self._pv_data_cache and weather in self._pv_data_cache[building]:
                        df = self._pv_data_cache[building][weather]
                        if "pv_generation_kw" in df.columns:
                            pv_gen = df["pv_generation_kw"].sum()
                            # Use PV carbon offset as approximation
                            pv_carbon = pv_gen * GRID_EMISSION_FACTOR
                        if "demand_kw" in df.columns:
                            demand = df["demand_kw"].sum()
                            baseline_carbon = demand * GRID_EMISSION_FACTOR

                carbon_data[building] = {
                    "baseline_kg": baseline_carbon,
                    "pv_kg": pv_carbon,
                }
                logger.debug(f"S07 carbon data for {building}: {carbon_data[building]}")

            if carbon_data:
                fig_carbon = CarbonReductionFigure(self.config)
                fig = fig_carbon.plot(carbon_data)
                fig_carbon.save(fig, "S07_carbon_reduction", self.output_dir)
                fig_carbon.close(fig)
                logger.info("Generated S7: Carbon reduction")
        except Exception as e:
            logger.error(f"Failed S7: {e}")

        # S8: Performance radar
        try:
            performance_data: dict[str, dict[str, float]] = {}
            for building in ALL_BUILDINGS:
                weather = "TMY"
                metrics = self._metrics_cache.get(building, {}).get(weather, {})

                # Get EUI reduction
                baseline_eui = 0
                pv_eui = 0
                if "baseline" in self._result_cache:
                    if building in self._result_cache["baseline"]:
                        if weather in self._result_cache["baseline"][building]:
                            baseline_eui = getattr(
                                self._result_cache["baseline"][building][weather],
                                "total_site_eui", 0
                            ) or 0
                if "pv" in self._result_cache:
                    if building in self._result_cache["pv"]:
                        if weather in self._result_cache["pv"][building]:
                            pv_eui = getattr(
                                self._result_cache["pv"][building][weather],
                                "total_site_eui", 0
                            ) or 0

                eui_reduction = ((baseline_eui - pv_eui) / baseline_eui * 100) if baseline_eui > 0 else 0

                sc = metrics.get("self_consumption", {})
                pr = metrics.get("peak_reduction", {})
                carbon = metrics.get("carbon", {})

                performance_data[building] = {
                    "EUI Reduction": min(eui_reduction, 100),
                    "Self-Consumption": sc.get("scr", 0),
                    "Self-Sufficiency": sc.get("ssr", 0),
                    "Peak Reduction": pr.get("peak_reduction_pct", 0),
                    "Carbon Savings": carbon.get("carbon_reduction_pct", 0),
                    "Cost Savings": eui_reduction * 0.8,  # Approximate
                }

            if performance_data:
                fig_radar = PerformanceRadarFigure(self.config)
                fig = fig_radar.plot(performance_data)
                fig_radar.save(fig, "S08_performance_radar", self.output_dir)
                fig_radar.close(fig)
                logger.info("Generated S8: Performance radar")
        except Exception as e:
            logger.error(f"Failed S8: {e}")


def main():
    """Main entry point for visualization generation."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    # Determine paths
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / "output"
    output_dir = base_path / "visualization" / "figures"

    logger.info("=" * 60)
    logger.info("PV Simulation Visualization Report Generator")
    logger.info("=" * 60)
    logger.info(f"Base path: {base_path}")
    logger.info(f"Output dir: {output_dir}")

    # Generate report
    report = VisualizationReport(base_path, output_dir)
    report.generate_all()

    logger.info("=" * 60)
    logger.info(f"All figures saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
