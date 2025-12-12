"""
Metrics calculator for PV simulation analysis.

Calculates:
- Self-Consumption Rate (SCR)
- Self-Sufficiency Rate (SSR)
- Curtailment (弃电量)
- Peak load reduction
- Storage utilization
"""

from typing import Any

import numpy as np
import pandas as pd

from .config import STORAGE_CAPACITY, GRID_EMISSION_FACTOR


class MetricsCalculator:
    """
    Calculator for PV system performance metrics.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def calc_self_consumption(
        self, df: pd.DataFrame
    ) -> dict[str, float]:
        """
        Calculate self-consumption metrics.

        Self-Consumption Rate (SCR): % of PV generation used on-site
        Self-Sufficiency Rate (SSR): % of demand met by PV

        Args:
            df: Processed hourly DataFrame with pv_generation_kw and demand_kw

        Returns:
            Dict with SCR, SSR, and energy values
        """
        if "pv_generation_kw" not in df.columns or "demand_kw" not in df.columns:
            return {
                "scr": 0.0,
                "ssr": 0.0,
                "self_consumed_kwh": 0.0,
                "total_pv_kwh": 0.0,
                "total_demand_kwh": 0.0,
            }

        # Total PV generation (kW * 1h = kWh)
        total_pv = df["pv_generation_kw"].sum()

        # Total demand
        total_demand = df["demand_kw"].sum()

        # Self-consumed energy: min(PV, demand) at each hour
        self_consumed = np.minimum(
            df["pv_generation_kw"].values,
            df["demand_kw"].values
        ).sum()

        # Include storage discharge as additional self-consumption
        if "storage_discharge_power_kw" in df.columns:
            # Storage discharge covers additional demand
            storage_discharge = df["storage_discharge_power_kw"].sum()
            self_consumed = min(self_consumed + storage_discharge, total_demand)

        # Self-Consumption Rate
        scr = (self_consumed / total_pv * 100) if total_pv > 0 else 0.0

        # Self-Sufficiency Rate
        ssr = (self_consumed / total_demand * 100) if total_demand > 0 else 0.0

        return {
            "scr": scr,
            "ssr": ssr,
            "self_consumed_kwh": self_consumed,
            "total_pv_kwh": total_pv,
            "total_demand_kwh": total_demand,
        }

    def calc_curtailment(
        self, df: pd.DataFrame, building: str
    ) -> dict[str, Any]:
        """
        Calculate curtailment (弃电量).

        Curtailment occurs when:
        - PV generation exceeds demand
        - AND storage is full (for buildings with storage)
        - AND no grid export is allowed

        Args:
            df: Processed hourly DataFrame
            building: Building type (for storage capacity)

        Returns:
            Dict with curtailment metrics and hourly series
        """
        if "pv_generation_kw" not in df.columns or "demand_kw" not in df.columns:
            return {
                "total_curtailed_kwh": 0.0,
                "curtailment_rate": 0.0,
                "curtailment_hours": 0,
                "hourly_curtailment": pd.Series(dtype=float),
            }

        capacity_kwh = STORAGE_CAPACITY.get(building, 0)

        # Excess PV (generation - demand)
        excess_pv = (df["pv_generation_kw"] - df["demand_kw"]).clip(lower=0)

        if capacity_kwh > 0 and "storage_charge_power_kw" in df.columns:
            # For buildings with storage:
            # Curtailment = excess - what can be stored
            # When SOC is high and charge power is limited, excess is curtailed

            # Get storage charge power (what's actually being stored)
            charge_power = df["storage_charge_power_kw"].fillna(0)

            # Curtailment = excess that couldn't be stored
            curtailment = (excess_pv - charge_power).clip(lower=0)

            # Also check if storage is full (SOC near 100%)
            if "storage_soc_pct" in df.columns:
                # When SOC is 100%, all excess is curtailed
                full_storage_mask = df["storage_soc_pct"] >= 99.9
                curtailment = curtailment.where(
                    ~full_storage_mask | (curtailment > 0),
                    excess_pv
                )
        else:
            # For buildings without storage: all excess is curtailed
            curtailment = excess_pv

        total_curtailed = curtailment.sum()
        total_pv = df["pv_generation_kw"].sum()

        return {
            "total_curtailed_kwh": total_curtailed,
            "curtailment_rate": (total_curtailed / total_pv * 100) if total_pv > 0 else 0.0,
            "curtailment_hours": (curtailment > 0).sum(),
            "hourly_curtailment": curtailment,
            "monthly_curtailment": curtailment.groupby(df["month"]).sum() if "month" in df.columns else None,
        }

    def calc_peak_reduction(
        self, baseline_df: pd.DataFrame, pv_df: pd.DataFrame
    ) -> dict[str, float]:
        """
        Calculate peak load reduction.

        Args:
            baseline_df: Baseline hourly data
            pv_df: PV simulation hourly data

        Returns:
            Dict with peak reduction metrics
        """
        # Get demand columns
        baseline_demand = None
        pv_net_demand = None

        if "demand_kw" in baseline_df.columns:
            baseline_demand = baseline_df["demand_kw"]

        if "net_demand_kw" in pv_df.columns:
            pv_net_demand = pv_df["net_demand_kw"]
        elif "demand_kw" in pv_df.columns:
            pv_net_demand = pv_df["demand_kw"]

        if baseline_demand is None or pv_net_demand is None:
            return {
                "baseline_peak_kw": 0.0,
                "pv_peak_kw": 0.0,
                "peak_reduction_kw": 0.0,
                "peak_reduction_pct": 0.0,
            }

        baseline_peak = baseline_demand.max()
        pv_peak = pv_net_demand.max()
        reduction = baseline_peak - pv_peak

        return {
            "baseline_peak_kw": baseline_peak,
            "pv_peak_kw": pv_peak,
            "peak_reduction_kw": reduction,
            "peak_reduction_pct": (reduction / baseline_peak * 100) if baseline_peak > 0 else 0.0,
        }

    def calc_storage_metrics(
        self, df: pd.DataFrame, building: str
    ) -> dict[str, float]:
        """
        Calculate storage system metrics.

        Args:
            df: Processed hourly DataFrame
            building: Building type

        Returns:
            Dict with storage metrics
        """
        capacity_kwh = STORAGE_CAPACITY.get(building, 0)

        if capacity_kwh == 0:
            return {
                "capacity_kwh": 0.0,
                "total_charged_kwh": 0.0,
                "total_discharged_kwh": 0.0,
                "avg_soc_pct": 0.0,
                "full_cycles": 0.0,
                "utilization_rate": 0.0,
            }

        # Total energy charged/discharged
        total_charged = df["storage_charge_power_kw"].sum() if "storage_charge_power_kw" in df.columns else 0
        total_discharged = df["storage_discharge_power_kw"].sum() if "storage_discharge_power_kw" in df.columns else 0

        # Average SOC
        avg_soc = df["storage_soc_pct"].mean() if "storage_soc_pct" in df.columns else 0

        # Full equivalent cycles
        full_cycles = total_discharged / capacity_kwh if capacity_kwh > 0 else 0

        # Utilization rate (hours with charge/discharge activity)
        active_hours = 0
        if "storage_charge_power_kw" in df.columns and "storage_discharge_power_kw" in df.columns:
            active_hours = ((df["storage_charge_power_kw"] > 0) | (df["storage_discharge_power_kw"] > 0)).sum()

        utilization_rate = active_hours / len(df) * 100 if len(df) > 0 else 0

        return {
            "capacity_kwh": capacity_kwh,
            "total_charged_kwh": total_charged,
            "total_discharged_kwh": total_discharged,
            "avg_soc_pct": avg_soc,
            "full_cycles": full_cycles,
            "utilization_rate": utilization_rate,
        }

    def calc_carbon_reduction(
        self, df: pd.DataFrame, baseline_df: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        Calculate carbon emission reduction.

        Args:
            df: PV simulation hourly data
            baseline_df: Optional baseline data for comparison

        Returns:
            Dict with carbon metrics (in kgCO2)
        """
        # PV generation offset
        pv_generation = df["pv_generation_kw"].sum() if "pv_generation_kw" in df.columns else 0
        pv_carbon_offset = pv_generation * GRID_EMISSION_FACTOR

        # If baseline provided, calculate reduction
        baseline_carbon = 0
        pv_carbon = 0

        if baseline_df is not None and "demand_kw" in baseline_df.columns:
            baseline_energy = baseline_df["demand_kw"].sum()
            baseline_carbon = baseline_energy * GRID_EMISSION_FACTOR

        if "net_demand_kw" in df.columns:
            net_energy = df["net_demand_kw"].clip(lower=0).sum()
            pv_carbon = net_energy * GRID_EMISSION_FACTOR

        carbon_reduction = baseline_carbon - pv_carbon if baseline_carbon > 0 else pv_carbon_offset

        return {
            "pv_carbon_offset_kg": pv_carbon_offset,
            "baseline_carbon_kg": baseline_carbon,
            "pv_system_carbon_kg": pv_carbon,
            "carbon_reduction_kg": carbon_reduction,
            "carbon_reduction_pct": (carbon_reduction / baseline_carbon * 100) if baseline_carbon > 0 else 0.0,
        }

    def calc_eui_comparison(
        self, baseline_result: Any, optimization_result: Any, pv_result: Any
    ) -> dict[str, float]:
        """
        Calculate EUI comparison across simulation stages.

        Args:
            baseline_result: Baseline SimulationResult
            optimization_result: Optimization SimulationResult
            pv_result: PV SimulationResult

        Returns:
            Dict with EUI values and changes
        """
        baseline_eui = getattr(baseline_result, "total_site_eui", 0) or 0
        optimization_eui = getattr(optimization_result, "total_site_eui", 0) or 0
        pv_eui = getattr(pv_result, "total_site_eui", 0) or 0

        # Calculate changes
        ecm_reduction = baseline_eui - optimization_eui
        pv_reduction = optimization_eui - pv_eui
        total_reduction = baseline_eui - pv_eui

        return {
            "baseline_eui": baseline_eui,
            "optimization_eui": optimization_eui,
            "pv_eui": pv_eui,
            "ecm_reduction": ecm_reduction,
            "ecm_reduction_pct": (ecm_reduction / baseline_eui * 100) if baseline_eui > 0 else 0.0,
            "pv_reduction": pv_reduction,
            "pv_reduction_pct": (pv_reduction / optimization_eui * 100) if optimization_eui > 0 else 0.0,
            "total_reduction": total_reduction,
            "total_reduction_pct": (total_reduction / baseline_eui * 100) if baseline_eui > 0 else 0.0,
        }

    def calc_pv_efficiency(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate PV system efficiency metrics.

        Args:
            df: Processed hourly DataFrame

        Returns:
            Dict with efficiency metrics
        """
        if "pv_generation_kw" not in df.columns or "outdoor_temp_c" not in df.columns:
            return {
                "avg_generation_kw": 0.0,
                "peak_generation_kw": 0.0,
                "generation_hours": 0,
                "capacity_factor": 0.0,
            }

        # Filter hours with generation
        generation_mask = df["pv_generation_kw"] > 0
        generation_hours = generation_mask.sum()

        avg_generation = df.loc[generation_mask, "pv_generation_kw"].mean() if generation_hours > 0 else 0
        peak_generation = df["pv_generation_kw"].max()

        # Capacity factor (actual / theoretical max)
        # Assuming peak generation represents installed capacity
        capacity_factor = (df["pv_generation_kw"].sum() / (peak_generation * len(df)) * 100) if peak_generation > 0 else 0

        # Temperature correlation
        if generation_hours > 10:
            temp_corr = df.loc[generation_mask, ["pv_generation_kw", "outdoor_temp_c"]].corr().iloc[0, 1]
        else:
            temp_corr = 0.0

        return {
            "avg_generation_kw": avg_generation,
            "peak_generation_kw": peak_generation,
            "generation_hours": generation_hours,
            "capacity_factor": capacity_factor,
            "temp_correlation": temp_corr,
        }

    def calc_all_metrics(
        self, df: pd.DataFrame, building: str, baseline_df: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """
        Calculate all metrics for a building-weather combination.

        Args:
            df: Processed PV hourly DataFrame
            building: Building type
            baseline_df: Optional baseline DataFrame

        Returns:
            Dict with all metrics
        """
        metrics = {}

        # Self-consumption
        metrics["self_consumption"] = self.calc_self_consumption(df)

        # Curtailment
        metrics["curtailment"] = self.calc_curtailment(df, building)

        # Storage
        metrics["storage"] = self.calc_storage_metrics(df, building)

        # Carbon
        metrics["carbon"] = self.calc_carbon_reduction(df, baseline_df)

        # PV efficiency
        metrics["pv_efficiency"] = self.calc_pv_efficiency(df)

        # Peak reduction
        if baseline_df is not None:
            metrics["peak_reduction"] = self.calc_peak_reduction(baseline_df, df)

        return metrics
