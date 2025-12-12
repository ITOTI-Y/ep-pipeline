"""
Data loader for PV simulation results.

Handles loading of:
- PV hourly CSV data
- Baseline/Optimization/PV result.pkl files
- ECM sampling results CSV
"""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ALL_BUILDINGS, ALL_CLIMATES


class DataLoader:
    """
    Data loader for simulation results.

    Supports loading from various output directories:
    - baseline/
    - optimization/
    - pv/
    - ecm/
    """

    def __init__(self, base_path: Path | str | None = None):
        """
        Initialize data loader.

        Args:
            base_path: Base path to output directory. Defaults to backend/output.
        """
        if base_path is None:
            # Default to backend/output relative to this file
            self.base_path = Path(__file__).parent.parent / "output"
        else:
            self.base_path = Path(base_path)

    def load_pv_hourly(self, building: str, weather: str) -> pd.DataFrame:
        """
        Load PV hourly output CSV data.

        Args:
            building: Building type (e.g., 'SingleFamilyResidential')
            weather: Weather scenario (e.g., 'TMY', 'SSP585')

        Returns:
            DataFrame with hourly PV simulation data
        """
        csv_path = self.base_path / "pv" / building / weather / "pv_out.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"PV output file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        return df

    def load_baseline_hourly(self, building: str, weather: str) -> pd.DataFrame:
        """
        Load baseline hourly output CSV data.

        Args:
            building: Building type
            weather: Weather scenario

        Returns:
            DataFrame with hourly baseline simulation data
        """
        csv_path = self.base_path / "baseline" / building / weather / "baseline_out.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Baseline output file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        return df

    def load_result_pkl(
        self, result_type: str, building: str, weather: str
    ) -> Any:
        """
        Load simulation result from pickle file.

        Args:
            result_type: Type of result ('baseline', 'optimization', 'pv')
            building: Building type
            weather: Weather scenario

        Returns:
            SimulationResult object
        """
        pkl_path = self.base_path / result_type / building / weather / "result.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(f"Result file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            result = pickle.load(f)

        return result

    def load_baseline_result(self, building: str, weather: str) -> Any:
        """Load baseline simulation result."""
        return self.load_result_pkl("baseline", building, weather)

    def load_optimization_result(self, building: str, weather: str) -> Any:
        """Load optimization simulation result."""
        return self.load_result_pkl("optimization", building, weather)

    def load_pv_result(self, building: str, weather: str) -> Any:
        """Load PV simulation result."""
        return self.load_result_pkl("pv", building, weather)

    def load_ecm_results(self) -> pd.DataFrame:
        """
        Load ECM sampling results CSV.

        Returns:
            DataFrame with ECM parameters and EUI results
        """
        csv_path = self.base_path / "ecm" / "results.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"ECM results file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        return df

    def load_all_results(
        self, result_type: str = "pv"
    ) -> dict[str, dict[str, Any]]:
        """
        Load all simulation results for all building-weather combinations.

        Args:
            result_type: Type of result to load ('baseline', 'optimization', 'pv')

        Returns:
            Nested dict: {building: {weather: SimulationResult}}
        """
        results = {}

        for building in ALL_BUILDINGS:
            results[building] = {}
            for weather in ALL_CLIMATES:
                try:
                    result = self.load_result_pkl(result_type, building, weather)
                    results[building][weather] = result
                except FileNotFoundError:
                    # Skip missing files
                    pass

        return results

    def load_all_pv_hourly(self) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Load all PV hourly data for all building-weather combinations.

        Returns:
            Nested dict: {building: {weather: DataFrame}}
        """
        data = {}

        for building in ALL_BUILDINGS:
            data[building] = {}
            for weather in ALL_CLIMATES:
                try:
                    df = self.load_pv_hourly(building, weather)
                    data[building][weather] = df
                except FileNotFoundError:
                    # Skip missing files
                    pass

        return data

    def get_available_combinations(
        self, result_type: str = "pv"
    ) -> list[tuple[str, str]]:
        """
        Get list of available building-weather combinations.

        Args:
            result_type: Type of result to check

        Returns:
            List of (building, weather) tuples
        """
        combinations = []

        for building in ALL_BUILDINGS:
            for weather in ALL_CLIMATES:
                pkl_path = (
                    self.base_path / result_type / building / weather / "result.pkl"
                )
                if pkl_path.exists():
                    combinations.append((building, weather))

        return combinations

    def check_data_availability(self) -> dict[str, list[tuple[str, str]]]:
        """
        Check data availability for all result types.

        Returns:
            Dict mapping result type to available combinations
        """
        availability = {}

        for result_type in ["baseline", "optimization", "pv"]:
            availability[result_type] = self.get_available_combinations(result_type)

        # Check ECM results
        ecm_path = self.base_path / "ecm" / "results.csv"
        availability["ecm"] = [("all", "all")] if ecm_path.exists() else []

        return availability
