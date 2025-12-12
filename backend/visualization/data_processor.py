"""
Data processor for PV simulation results.

Handles:
- DateTime parsing and indexing
- PV column extraction
- Unit conversions (J→kWh, W→kW)
- Time aggregation (hourly, daily, monthly)
"""

import re
from typing import Any

import numpy as np
import pandas as pd

from .config import STORAGE_CAPACITY


class DataProcessor:
    """
    Data processor for simulation results.

    Transforms raw CSV data into analysis-ready format.
    """

    # Column name patterns
    PV_GENERATOR_PATTERN = re.compile(r".*Generator Produced DC Electricity Rate \[W\].*")
    STORAGE_CHARGE_STATE_PATTERN = re.compile(r".*Electric Storage Simple Charge State \[J\].*")
    STORAGE_CHARGE_POWER_PATTERN = re.compile(r".*Electric Storage Charge Power \[W\].*")
    STORAGE_DISCHARGE_POWER_PATTERN = re.compile(r".*Electric Storage Discharge Power \[W\].*")
    DEMAND_PATTERN = re.compile(r".*Facility Total Electricity Demand Rate \[W\].*")
    ENERGY_PATTERN = re.compile(r"Electricity:Facility \[J\].*")
    TEMPERATURE_PATTERN = re.compile(r".*Site Outdoor Air Drybulb Temperature \[C\].*")

    def __init__(self):
        """Initialize data processor."""
        pass

    def process_pv_data(
        self, df: pd.DataFrame, building: str, year: int = 2020
    ) -> pd.DataFrame:
        """
        Process PV hourly data with all transformations.

        Args:
            df: Raw PV output DataFrame
            building: Building type (for storage capacity lookup)
            year: Year for datetime index

        Returns:
            Processed DataFrame with standardized columns
        """
        # Create a copy
        df = df.copy()

        # Parse datetime
        df = self._parse_datetime(df, year)

        # Extract and rename columns
        df = self._extract_columns(df, building)

        # Calculate derived metrics
        df = self._calculate_derived(df, building)

        return df

    def _parse_datetime(self, df: pd.DataFrame, year: int = 2020) -> pd.DataFrame:
        """
        Parse Date/Time column and set as index.

        Args:
            df: DataFrame with Date/Time column
            year: Year to use for datetime

        Returns:
            DataFrame with datetime index
        """
        if "Date/Time" not in df.columns:
            return df

        # Parse the date/time string (format: " MM/DD  HH:MM:SS")
        def parse_dt(s: str) -> pd.Timestamp:
            s = s.strip()
            parts = s.split()
            if len(parts) >= 2:
                date_part = parts[0]
                time_part = parts[1]
                month, day = date_part.split("/")
                hour = int(time_part.split(":")[0])
                # Handle hour 24 as hour 0 of next day
                if hour == 24:
                    hour = 0
                return pd.Timestamp(
                    year=year, month=int(month), day=int(day), hour=hour
                )
            return pd.NaT

        df["datetime"] = df["Date/Time"].apply(parse_dt)
        df = df.set_index("datetime")
        df = df.drop(columns=["Date/Time"], errors="ignore")

        # Add time-related columns
        df["month"] = df.index.month
        df["day"] = df.index.day
        df["hour"] = df.index.hour
        df["dayofyear"] = df.index.dayofyear

        return df

    def _extract_columns(self, df: pd.DataFrame, building: str) -> pd.DataFrame:
        """
        Extract and rename relevant columns.

        Args:
            df: DataFrame with raw column names
            building: Building type

        Returns:
            DataFrame with standardized column names
        """
        result = pd.DataFrame(index=df.index)

        # Copy time columns
        for col in ["month", "day", "hour", "dayofyear"]:
            if col in df.columns:
                result[col] = df[col]

        # Extract PV generation columns
        pv_cols = [c for c in df.columns if self.PV_GENERATOR_PATTERN.match(c)]
        if pv_cols:
            # Sum all PV generators, convert W to kW
            result["pv_generation_kw"] = df[pv_cols].sum(axis=1) / 1000
            result["pv_generation_count"] = len(pv_cols)

        # Extract storage columns
        charge_state_cols = [c for c in df.columns if self.STORAGE_CHARGE_STATE_PATTERN.match(c)]
        if charge_state_cols:
            # Convert J to kWh (1 kWh = 3,600,000 J)
            result["storage_charge_state_kwh"] = df[charge_state_cols[0]] / 3600000

        charge_power_cols = [c for c in df.columns if self.STORAGE_CHARGE_POWER_PATTERN.match(c)]
        if charge_power_cols:
            result["storage_charge_power_kw"] = df[charge_power_cols[0]] / 1000

        discharge_power_cols = [c for c in df.columns if self.STORAGE_DISCHARGE_POWER_PATTERN.match(c)]
        if discharge_power_cols:
            result["storage_discharge_power_kw"] = df[discharge_power_cols[0]] / 1000

        # Extract demand column
        demand_cols = [c for c in df.columns if self.DEMAND_PATTERN.match(c)]
        if demand_cols:
            result["demand_kw"] = df[demand_cols[0]] / 1000

        # Extract total energy column
        energy_cols = [c for c in df.columns if self.ENERGY_PATTERN.match(c)]
        if energy_cols:
            # Convert J to kWh
            result["total_energy_kwh"] = df[energy_cols[0]] / 3600000

        # Extract outdoor temperature
        temp_cols = [c for c in df.columns if self.TEMPERATURE_PATTERN.match(c)]
        if temp_cols:
            result["outdoor_temp_c"] = df[temp_cols[0]]

        return result

    def _calculate_derived(self, df: pd.DataFrame, building: str) -> pd.DataFrame:
        """
        Calculate derived metrics.

        Args:
            df: DataFrame with extracted columns
            building: Building type

        Returns:
            DataFrame with derived metrics
        """
        # Get storage capacity
        capacity_kwh = STORAGE_CAPACITY.get(building, 0)

        # Calculate SOC percentage
        if "storage_charge_state_kwh" in df.columns and capacity_kwh > 0:
            df["storage_soc_pct"] = (df["storage_charge_state_kwh"] / capacity_kwh) * 100
            df["storage_soc_pct"] = df["storage_soc_pct"].clip(0, 100)
        else:
            df["storage_soc_pct"] = 0.0

        # Calculate excess PV (generation - demand)
        if "pv_generation_kw" in df.columns and "demand_kw" in df.columns:
            df["excess_pv_kw"] = (df["pv_generation_kw"] - df["demand_kw"]).clip(lower=0)
            df["deficit_kw"] = (df["demand_kw"] - df["pv_generation_kw"]).clip(lower=0)

        # Calculate net demand (after PV and storage)
        if "demand_kw" in df.columns:
            net_demand = df["demand_kw"].copy()

            if "pv_generation_kw" in df.columns:
                net_demand = net_demand - df["pv_generation_kw"]

            if "storage_discharge_power_kw" in df.columns:
                net_demand = net_demand - df["storage_discharge_power_kw"]

            if "storage_charge_power_kw" in df.columns:
                net_demand = net_demand + df["storage_charge_power_kw"]

            df["net_demand_kw"] = net_demand

        return df

    def aggregate_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data to monthly level.

        Args:
            df: Hourly DataFrame with datetime index

        Returns:
            Monthly aggregated DataFrame
        """
        # Define aggregation rules
        agg_rules = {}

        # Sum for energy/power columns (kW * 1h = kWh)
        sum_cols = [
            "pv_generation_kw",
            "demand_kw",
            "excess_pv_kw",
            "deficit_kw",
            "storage_charge_power_kw",
            "storage_discharge_power_kw",
            "total_energy_kwh",
        ]
        for col in sum_cols:
            if col in df.columns:
                agg_rules[col] = "sum"

        # Mean for temperature and SOC
        mean_cols = ["outdoor_temp_c", "storage_soc_pct"]
        for col in mean_cols:
            if col in df.columns:
                agg_rules[col] = "mean"

        # Max for peak values
        if "demand_kw" in df.columns:
            agg_rules["peak_demand_kw"] = ("demand_kw", "max")

        if "pv_generation_kw" in df.columns:
            agg_rules["peak_pv_kw"] = ("pv_generation_kw", "max")

        # Group by month
        monthly = df.groupby("month").agg(agg_rules)

        return monthly

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data to daily level.

        Args:
            df: Hourly DataFrame with datetime index

        Returns:
            Daily aggregated DataFrame
        """
        # Define aggregation rules
        agg_rules = {}

        sum_cols = [
            "pv_generation_kw",
            "demand_kw",
            "excess_pv_kw",
            "deficit_kw",
            "storage_charge_power_kw",
            "storage_discharge_power_kw",
        ]
        for col in sum_cols:
            if col in df.columns:
                agg_rules[col] = "sum"

        mean_cols = ["outdoor_temp_c", "storage_soc_pct"]
        for col in mean_cols:
            if col in df.columns:
                agg_rules[col] = "mean"

        # Group by day of year
        daily = df.groupby("dayofyear").agg(agg_rules)

        return daily

    def get_typical_day(
        self, df: pd.DataFrame, month: int, day: int | None = None
    ) -> pd.DataFrame:
        """
        Get data for a typical day.

        Args:
            df: Hourly DataFrame
            month: Month to select
            day: Specific day (if None, averages across month)

        Returns:
            24-hour DataFrame
        """
        if day is not None:
            # Get specific day
            mask = (df["month"] == month) & (df["day"] == day)
            return df[mask].copy()
        else:
            # Average across month
            monthly_data = df[df["month"] == month]
            return monthly_data.groupby("hour").mean()

    def get_summer_winter_days(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get typical summer and winter days.

        Args:
            df: Hourly DataFrame

        Returns:
            Tuple of (summer_day, winter_day) DataFrames
        """
        # Summer: July 15
        summer_day = self.get_typical_day(df, month=7, day=15)

        # Winter: January 15
        winter_day = self.get_typical_day(df, month=1, day=15)

        return summer_day, winter_day


def extract_eui_summary(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Extract EUI summary from all results.

    Args:
        results: Nested dict of SimulationResult objects

    Returns:
        DataFrame with EUI values for all combinations
    """
    records = []

    for building, weathers in results.items():
        for weather, result in weathers.items():
            record = {
                "building": building,
                "weather": weather,
                "total_site_eui": getattr(result, "total_site_eui", None),
                "net_site_eui": getattr(result, "net_site_eui", None),
                "total_source_eui": getattr(result, "total_source_eui", None),
                "net_source_eui": getattr(result, "net_source_eui", None),
                "total_site_energy": getattr(result, "total_site_energy", None),
                "net_site_energy": getattr(result, "net_site_energy", None),
            }
            records.append(record)

    return pd.DataFrame(records)
