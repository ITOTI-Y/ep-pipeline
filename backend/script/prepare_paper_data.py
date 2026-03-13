"""
Prepare structured CSV data for the building carbon neutrality paper.

Generates 7 CSV files from EnergyPlus simulation outputs, XGBoost evaluation,
and Cambium 2024 grid data.

Usage: uv run python -m backend.script.prepare_paper_data
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from backend.models.enums import BuildingType
from backend.utils.config.config_manager import ConfigManager

# ── Constants ────────────────────────────────────────────────────────────────

BUILDING_TYPES = [bt.value for bt in BuildingType]
WEATHER_CODES = ["TMY", "SSP126", "SSP245", "SSP370", "SSP434", "SSP585"]
STAGES = ["baseline", "ecm", "pv"]

# eGRID RFCW emission factors
EF_ELEC_AVG = 0.4134  # kg CO₂/kWh (average)
EF_ELEC_MARGINAL = 0.7973  # kg CO₂/kWh (marginal)
CURRENT_RE = 0.076  # current RE fraction 7.6%

# EPA natural gas emission factor (direct combustion)
EF_GAS = 0.181  # kg CO₂/kWh

# Mode C renewable energy penetration levels
R_VALUES = [0.0, 0.076, 0.15, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]

# Cambium 2024
CAMBIUM_SCENARIOS = ["MidCase", "LowRECost", "HighRECost_LowNGPrice"]
CAMBIUM_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]

# ── Path helpers ─────────────────────────────────────────────────────────────

CONFIG = ConfigManager()
DATA_DIR = CONFIG.paths.data_dir
PAPER_DIR = CONFIG.paths.csv_dir

# Stage → (config path, sql filename)
STAGE_SQL_MAP: dict[str, tuple[Path, str]] = {
    "baseline": (CONFIG.paths.baseline_dir, "baseline_out.sql"),
    "ecm": (CONFIG.paths.optimization_dir, "optimization_out.sql"),
    "pv": (CONFIG.paths.pv_dir, "pv_out.sql"),
}


def sql_path(stage: str, building: str, weather: str) -> Path:
    stage_dir, fname = STAGE_SQL_MAP[stage]
    return stage_dir / building / weather / fname


def cambium_path(scenario: str, year: int) -> Path:
    return DATA_DIR / "cambium" / f"Cambium24_{scenario}_hourly_PJM_West_{year}.csv"


def _build_area_map(df_energy: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Build (building_type, weather_code) → total_building_area lookup from baseline rows."""
    bl = df_energy[df_energy["stage"] == "baseline"]
    return {
        (row["building_type"], row["weather_code"]): row["total_building_area"]
        for _, row in bl.iterrows()
    }


# ── SQL helpers ──────────────────────────────────────────────────────────────


def query_end_uses(db_path: Path) -> dict[str, float]:
    """Extract annual electricity and natural gas from End Uses table."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT ColumnName, CAST(Value AS REAL) FROM TabularDataWithStrings "
            "WHERE ReportName='AnnualBuildingUtilityPerformanceSummary' "
            "AND TableName='End Uses' AND RowName='Total End Uses' AND Units='kWh'"
        ).fetchall()
    return {r[0].strip(): r[1] for r in rows}


def query_electric_loads(db_path: Path) -> dict[str, float]:
    """Extract Electric Loads Satisfied table values."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT RowName, CAST(Value AS REAL) FROM TabularDataWithStrings "
            "WHERE ReportName='AnnualBuildingUtilityPerformanceSummary' "
            "AND TableName='Electric Loads Satisfied' "
            "AND ColumnName='Electricity' AND Units='kWh'"
        ).fetchall()
    return {r[0].strip(): r[1] for r in rows if r[0].strip()}


def query_hourly_power(db_path: Path) -> pd.DataFrame:
    """Extract 8760 hourly power data from ReportVariableWithTime.

    ElectricityNet:Facility only exists in some buildings. For those without it,
    compute net grid exchange from: demand - PV + charge - discharge (verified
    to match ElectricityNet:Facility where both exist).
    """
    with sqlite3.connect(db_path) as conn:
        # Check if ElectricityNet:Facility exists
        has_net = (
            conn.execute(
                "SELECT count(*) FROM ReportVariableWithTime "
                "WHERE Name='ElectricityNet:Facility'"
            ).fetchone()[0]
            > 0
        )

        target_vars = [
            "Facility Total Electricity Demand Rate",
            "Generator Produced DC Electricity Rate",
            "Electric Storage Charge Power",
            "Electric Storage Discharge Power",
            "Electric Storage Simple Charge State",
        ]
        if has_net:
            target_vars.append("ElectricityNet:Facility")

        placeholders = ",".join(f"'{v}'" for v in target_vars)
        net_col = (
            "max(CASE WHEN Name='ElectricityNet:Facility' THEN Value END) as net_j,"
            if has_net
            else ""
        )

        df = pd.read_sql_query(
            f"""
            SELECT Month, Day, Hour,
                max(CASE WHEN Name='Facility Total Electricity Demand Rate'
                    THEN Value ELSE 0 END) as demand_w,
                sum(CASE WHEN Name='Generator Produced DC Electricity Rate'
                    THEN Value ELSE 0 END) as pv_w,
                {net_col}
                max(CASE WHEN Name='Electric Storage Charge Power'
                    THEN Value ELSE 0 END) as charge_w,
                max(CASE WHEN Name='Electric Storage Discharge Power'
                    THEN Value ELSE 0 END) as discharge_w,
                max(CASE WHEN Name='Electric Storage Simple Charge State'
                    THEN Value ELSE 0 END) as soc_j
            FROM ReportVariableWithTime
            WHERE Name IN ({placeholders})
            GROUP BY Month, Day, Hour
            ORDER BY Month, Day, Hour
            """,
            conn,
        )

    # Compute net grid exchange in J
    if has_net:
        df["net_j"] = pd.to_numeric(df["net_j"], errors="coerce").fillna(0.0)
    else:
        # Derive from components: net_J = (demand - PV + charge - discharge) x 3600
        df["net_j"] = (
            df["demand_w"] - df["pv_w"] + df["charge_w"] - df["discharge_w"]
        ) * 3600

    # Convert units
    df["demand_kw"] = df["demand_w"] / 1000
    df["pv_generation_kw"] = df["pv_w"] / 1000
    df["purchased_kw"] = df["net_j"].clip(lower=0) / 3_600_000
    df["exported_kw"] = (-df["net_j"]).clip(lower=0) / 3_600_000
    df["storage_charge_kw"] = df["charge_w"] / 1000
    df["storage_discharge_kw"] = df["discharge_w"] / 1000
    df["storage_soc_kwh"] = df["soc_j"] / 3_600_000

    return df[
        [
            "Month",
            "Day",
            "Hour",
            "demand_kw",
            "pv_generation_kw",
            "purchased_kw",
            "exported_kw",
            "storage_charge_kw",
            "storage_discharge_kw",
            "storage_soc_kwh",
        ]
    ].rename(columns={"Month": "month", "Day": "day", "Hour": "hour"})


def query_baseline_hourly_demand(db_path: Path) -> np.ndarray:
    """Extract 8760 hourly electricity demand (kWh/h) for baseline/ecm (no PV).

    For stages without PV/storage, all demand = purchased from grid, export = 0.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT Month, Day, Hour,
                max(CASE WHEN Name='Facility Total Electricity Demand Rate'
                    THEN Value ELSE 0 END) as demand_w
            FROM ReportVariableWithTime
            WHERE Name = 'Facility Total Electricity Demand Rate'
            GROUP BY Month, Day, Hour
            ORDER BY Month, Day, Hour
            """,
            conn,
        )
    assert len(df) == 8760, f"Expected 8760 hours, got {len(df)} from {db_path}"
    return df["demand_w"].values / 1000  # W → kW = kWh/h


# ── CSV 1: Energy Summary ───────────────────────────────────────────────────


def prepare_energy_summary() -> pd.DataFrame:
    """Prepare 90-row energy summary from Results_data.csv."""
    logger.info("Preparing CSV 1: Energy Summary")
    src = CONFIG.paths.visualization_dir / "Results_data.csv"
    df = pd.read_csv(src)

    # Map data_type to stage names
    stage_map = {"baseline": "baseline", "optimization": "ecm", "pv": "pv"}
    df["stage"] = df["data_type"].map(stage_map)

    out = df[
        [
            "building_type",
            "weather_code",
            "stage",
            "total_building_area",
            "total_site_energy",
            "total_source_energy",
            "net_site_energy",
            "net_source_energy",
            "total_site_eui",
            "total_source_eui",
            "net_site_eui",
            "net_source_eui",
            "predicted_eui",
        ]
    ].copy()

    # Compute ecm_improvement_pct (site EUI reduction vs baseline)
    ecm_improve = []
    pv_reduce = []
    for _, row in out.iterrows():
        bt, wc, stage = row["building_type"], row["weather_code"], row["stage"]
        if stage == "ecm":
            bl = out[
                (out["building_type"] == bt)
                & (out["weather_code"] == wc)
                & (out["stage"] == "baseline")
            ]
            if not bl.empty:
                bl_eui = bl.iloc[0]["total_site_eui"]
                ecm_improve.append(
                    (bl_eui - row["total_site_eui"]) / bl_eui * 100
                    if bl_eui
                    else np.nan
                )
            else:
                ecm_improve.append(np.nan)
        else:
            ecm_improve.append(np.nan)

        if stage == "pv":
            ecm_row = out[
                (out["building_type"] == bt)
                & (out["weather_code"] == wc)
                & (out["stage"] == "ecm")
            ]
            if not ecm_row.empty:
                ecm_eui = ecm_row.iloc[0]["total_site_eui"]
                pv_reduce.append(
                    (ecm_eui - row["net_site_eui"]) / ecm_eui * 100
                    if ecm_eui
                    else np.nan
                )
            else:
                pv_reduce.append(np.nan)
        else:
            pv_reduce.append(np.nan)

    out["ecm_improvement_pct"] = ecm_improve
    out["pv_reduction_pct"] = pv_reduce

    logger.info(f"  → {len(out)} rows")
    return out


# ── CSV 2: Surrogate Evaluation ─────────────────────────────────────────────


def prepare_surrogate_evaluation() -> pd.DataFrame:
    """Prepare 5-row surrogate model evaluation from evaluate.json files."""
    logger.info("Preparing CSV 2: Surrogate Evaluation")
    rows = []
    for bt in BUILDING_TYPES:
        path = CONFIG.paths.optimization_dir / bt / "evaluate.json"
        with open(path) as f:
            d = json.load(f)
        row = {
            "building_type": bt,
            "overall_r2": d["r2"],
            "overall_rmse": d["rmse"],
            "overall_mae": d["mae"],
        }
        for i in range(1, 5):
            row[f"output_{i}_r2"] = d[f"output_{i}_r2_score"]
        # pred_error_pct: RMSE / mean observed EUI as proxy
        row["pred_error_pct"] = np.nan  # will compute from energy summary
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"  → {len(df)} rows")
    return df


def update_surrogate_pred_error(
    df_eval: pd.DataFrame, df_energy: pd.DataFrame
) -> pd.DataFrame:
    """Fill pred_error_pct using mean baseline source EUI."""
    for i, row in df_eval.iterrows():
        bt = row["building_type"]
        bl = df_energy[
            (df_energy["building_type"] == bt) & (df_energy["stage"] == "baseline")
        ]["total_source_eui"]
        if not bl.empty:
            mean_eui = bl.mean()
            df_eval.at[i, "pred_error_pct"] = row["overall_rmse"] / mean_eui * 100
    return df_eval


# ── CSV 3: Electricity Balance ──────────────────────────────────────────────


def prepare_electricity_balance() -> pd.DataFrame:
    """Prepare 90-row electricity balance from SQL databases."""
    logger.info("Preparing CSV 3: Electricity Balance")
    rows = []
    for bt in BUILDING_TYPES:
        for wc in WEATHER_CODES:
            for stage in STAGES:
                db = sql_path(stage, bt, wc)
                if not db.exists():
                    logger.warning(f"  Missing: {db}")
                    continue

                end_uses = query_end_uses(db)
                elec_loads = query_electric_loads(db)

                total_elec = end_uses.get("Electricity", 0.0)
                total_gas = end_uses.get("Natural Gas", 0.0)
                pv_gen = elec_loads.get("Photovoltaic Power", 0.0)
                purchased = elec_loads.get("Electricity Coming From Utility", 0.0)
                exported = elec_loads.get("Surplus Electricity Going To Utility", 0.0)
                net_from_utility = elec_loads.get("Net Electricity From Utility", 0.0)

                # SCR = (PV_gen - exported) / PV_gen
                scr = (
                    ((pv_gen - exported) / pv_gen * 100)
                    if (stage == "pv" and pv_gen > 0)
                    else np.nan
                )
                # SSR = (total_elec - purchased) / total_elec
                ssr = (
                    ((total_elec - purchased) / total_elec * 100)
                    if (stage == "pv" and total_elec > 0)
                    else np.nan
                )

                rows.append(
                    {
                        "building_type": bt,
                        "weather_code": wc,
                        "stage": stage,
                        "total_electricity_kwh": total_elec,
                        "total_natural_gas_kwh": total_gas,
                        "pv_generation_kwh": pv_gen if stage == "pv" else np.nan,
                        "purchased_electricity_kwh": purchased,
                        "exported_electricity_kwh": exported
                        if stage == "pv"
                        else np.nan,
                        "net_electricity_from_utility_kwh": net_from_utility,
                        "scr": scr,
                        "ssr": ssr,
                    }
                )

    df = pd.DataFrame(rows)
    logger.info(f"  → {len(df)} rows")
    return df


# ── CSV 4: Hourly Power ─────────────────────────────────────────────────────


def prepare_hourly_power() -> pd.DataFrame:
    """Prepare ~262800-row hourly power data from 30 PV SQL databases."""
    logger.info("Preparing CSV 4: Hourly Power")
    frames = []
    for bt in BUILDING_TYPES:
        for wc in WEATHER_CODES:
            db = sql_path("pv", bt, wc)
            if not db.exists():
                logger.warning(f"  Missing: {db}")
                continue

            df = query_hourly_power(db)
            df.insert(0, "building_type", bt)
            df.insert(1, "weather_code", wc)
            frames.append(df)
            logger.info(f"  {bt}/{wc}: {len(df)} hours")

    result = pd.concat(frames, ignore_index=True)
    logger.info(f"  → {len(result)} total rows")
    return result


# ── CSV 5: Carbon Mode B + C ────────────────────────────────────────────────


def prepare_carbon_mode_bc(
    df_elec: pd.DataFrame, df_energy: pd.DataFrame
) -> pd.DataFrame:
    """Prepare carbon Mode B (90 rows) + Mode C (330 rows)."""
    logger.info("Preparing CSV 5: Carbon Mode B + C")

    area_map = _build_area_map(df_energy)

    # --- Part A: Mode B (all 3 stages x 30 scenarios = 90 rows) ---
    mode_b_rows = []
    for _, r in df_elec.iterrows():
        bt, wc, stage = r["building_type"], r["weather_code"], r["stage"]
        purchased = r["purchased_electricity_kwh"]
        exported = (
            r["exported_electricity_kwh"]
            if not np.isnan(r.get("exported_electricity_kwh", np.nan))
            else 0.0
        )
        gas = r["total_natural_gas_kwh"]

        carbon_purchased = purchased * EF_ELEC_AVG
        carbon_credit = exported * EF_ELEC_MARGINAL
        carbon_gas = gas * EF_GAS
        carbon_net = carbon_purchased - carbon_credit + carbon_gas

        area = area_map[(bt, wc)]

        mode_b_rows.append(
            {
                "building_type": bt,
                "weather_code": wc,
                "stage": stage,
                "mode": "mode_b",
                "purchased_kwh": purchased,
                "exported_kwh": exported,
                "carbon_elec_purchased_kg": carbon_purchased,
                "carbon_elec_exported_credit_kg": carbon_credit,
                "carbon_gas_kg": carbon_gas,
                "carbon_net_kg": carbon_net,
                "carbon_intensity_kgm2": carbon_net / area,
            }
        )

    df_b = pd.DataFrame(mode_b_rows)

    # --- Part B: Mode C (pv stage only, 30 x 11 R = 330 rows) ---
    mode_c_rows = []
    pv_elec = df_elec[df_elec["stage"] == "pv"]
    for _, r in pv_elec.iterrows():
        bt, wc = r["building_type"], r["weather_code"]
        net_elec = r["net_electricity_from_utility_kwh"]
        gas = r["total_natural_gas_kwh"]

        area = area_map.get((bt, wc), 1.0)

        for r_val in R_VALUES:
            elec_intensity = max(0, net_elec) * EF_ELEC_AVG * (1 - r_val) / area
            gas_intensity = gas * EF_GAS / area
            mode_c_rows.append(
                {
                    "building_type": bt,
                    "weather_code": wc,
                    "stage": "pv",
                    "mode": "mode_c",
                    "R": r_val,
                    "carbon_elec_intensity_kgm2": elec_intensity,
                    "carbon_gas_intensity_kgm2": gas_intensity,
                    "carbon_total_intensity_kgm2": elec_intensity + gas_intensity,
                }
            )

    df_c = pd.DataFrame(mode_c_rows)

    result = pd.concat([df_b, df_c], ignore_index=True)
    logger.info(f"  → Mode B: {len(df_b)} rows, Mode C: {len(df_c)} rows")
    return result


# ── CSV 6: Carbon Mode A (Cambium hourly) ───────────────────────────────────


@dataclass
class CambiumFactors:
    """Pre-computed Cambium emission factors adjusted for distribution loss (kg/kWh)."""

    aer: np.ndarray  # 8760 hourly average emission rate
    lrmer: np.ndarray  # 8760 hourly long-run marginal emission rate
    aer_raw_mean: float  # raw aer_load_co2e annual mean (kg/MWh)
    lrmer_raw_mean: float  # raw lrmer_co2e annual mean (kg/MWh)


def load_cambium(scenario: str, year: int) -> CambiumFactors:
    """Load Cambium CSV and pre-compute distloss-adjusted emission factors."""
    path = cambium_path(scenario, year)
    df = pd.read_csv(  # ty: ignore[no-matching-overload]
        path,
        skiprows=5,
        usecols=["aer_load_co2e", "lrmer_co2e", "distloss_rate_avg"],
    )
    distloss = df["distloss_rate_avg"].values
    factor = 1 / (1 - distloss) / 1000  # kg/MWh → kg/kWh with distloss adj
    return CambiumFactors(
        aer=df["aer_load_co2e"].values * factor,
        lrmer=df["lrmer_co2e"].values * factor,
        aer_raw_mean=df["aer_load_co2e"].mean(),
        lrmer_raw_mean=df["lrmer_co2e"].mean(),
    )


def prepare_carbon_mode_a(
    df_hourly: pd.DataFrame, df_elec: pd.DataFrame, df_energy: pd.DataFrame
) -> pd.DataFrame:
    """Prepare Mode A carbon accounting using Cambium hourly emission factors.

    Processes both baseline and pv stages:
    - baseline: all demand = purchased, no export (query SQL directly)
    - pv: use 04_hourly_power data (purchased/exported from PV+storage dispatch)
    """
    logger.info("Preparing CSV 6: Carbon Mode A (Cambium)")

    # Pre-load all Cambium data (emission factors pre-adjusted for distloss)
    cambium_cache: dict[tuple[str, int], CambiumFactors] = {}
    for scen in CAMBIUM_SCENARIOS:
        for yr in CAMBIUM_YEARS:
            cambium_cache[(scen, yr)] = load_cambium(scen, yr)
    logger.info(f"  Loaded {len(cambium_cache)} Cambium files")

    # Pre-build area and gas lookups
    area_map = _build_area_map(df_energy)
    hourly_groups = dict(list(df_hourly.groupby(["building_type", "weather_code"])))

    rows = []
    for bt in BUILDING_TYPES:
        for wc in WEATHER_CODES:
            area = area_map.get((bt, wc), 1.0)

            # --- Baseline stage: query SQL directly ---
            bl_db = sql_path("baseline", bt, wc)
            if bl_db.exists():
                bl_demand_kwh = query_baseline_hourly_demand(bl_db)

                bl_gas_row = df_elec[
                    (df_elec["building_type"] == bt)
                    & (df_elec["weather_code"] == wc)
                    & (df_elec["stage"] == "baseline")
                ]
                bl_gas = (
                    bl_gas_row.iloc[0]["total_natural_gas_kwh"]
                    if not bl_gas_row.empty
                    else 0.0
                )

                for scen in CAMBIUM_SCENARIOS:
                    for yr in CAMBIUM_YEARS:
                        cam = cambium_cache[(scen, yr)]

                        carbon_purchased = (bl_demand_kwh * cam.aer).sum()
                        carbon_gas = bl_gas * EF_GAS
                        carbon_net = carbon_purchased + carbon_gas

                        rows.append(
                            {
                                "building_type": bt,
                                "weather_code": wc,
                                "stage": "baseline",
                                "cambium_scenario": scen,
                                "cambium_year": yr,
                                "aer_annual_mean": cam.aer_raw_mean,
                                "lrmer_annual_mean": cam.lrmer_raw_mean,
                                "carbon_elec_purchased_kg": carbon_purchased,
                                "carbon_elec_exported_credit_kg": 0.0,
                                "carbon_gas_kg": carbon_gas,
                                "carbon_net_kg": carbon_net,
                                "carbon_intensity_kgm2": carbon_net / area,
                            }
                        )
            else:
                logger.warning(f"  Missing baseline SQL: {bl_db}")

            # --- PV stage: use hourly power data ---
            hourly = hourly_groups.get((bt, wc))
            if hourly is None or hourly.empty:
                logger.warning(f"  No hourly data for {bt}/{wc}")
                continue

            assert len(hourly) == 8760, (
                f"Expected 8760 hours, got {len(hourly)} for {bt}/{wc}"
            )

            purchased_kwh = hourly["purchased_kw"].values
            exported_kwh = hourly["exported_kw"].values

            gas_row = df_elec[
                (df_elec["building_type"] == bt)
                & (df_elec["weather_code"] == wc)
                & (df_elec["stage"] == "pv")
            ]
            gas_annual = (
                gas_row.iloc[0]["total_natural_gas_kwh"] if not gas_row.empty else 0.0
            )

            for scen in CAMBIUM_SCENARIOS:
                for yr in CAMBIUM_YEARS:
                    cam = cambium_cache[(scen, yr)]

                    carbon_purchased = (purchased_kwh * cam.aer).sum()
                    carbon_credit = (exported_kwh * cam.lrmer).sum()
                    carbon_gas = gas_annual * EF_GAS
                    carbon_net = carbon_purchased - carbon_credit + carbon_gas

                    rows.append(
                        {
                            "building_type": bt,
                            "weather_code": wc,
                            "stage": "pv",
                            "cambium_scenario": scen,
                            "cambium_year": yr,
                            "aer_annual_mean": cam.aer_raw_mean,
                            "lrmer_annual_mean": cam.lrmer_raw_mean,
                            "carbon_elec_purchased_kg": carbon_purchased,
                            "carbon_elec_exported_credit_kg": carbon_credit,
                            "carbon_gas_kg": carbon_gas,
                            "carbon_net_kg": carbon_net,
                            "carbon_intensity_kgm2": carbon_net / area,
                        }
                    )

    df = pd.DataFrame(rows)
    n_bl = len(df[df["stage"] == "baseline"])
    n_pv = len(df[df["stage"] == "pv"])
    logger.info(f"  → {len(df)} rows (baseline: {n_bl}, pv: {n_pv})")
    return df


# ── CSV 7: BCRC Summary ─────────────────────────────────────────────────────


def prepare_bcrc_summary(
    df_energy: pd.DataFrame,
    df_carbon_bc: pd.DataFrame,
    df_carbon_a: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare 30-row BCRC summary."""
    logger.info("Preparing CSV 7: BCRC Summary")

    mode_b = df_carbon_bc[df_carbon_bc["mode"] == "mode_b"]
    mode_c = df_carbon_bc[df_carbon_bc["mode"] == "mode_c"]

    # Build indexed lookups for fast access
    energy_idx = df_energy.set_index(["building_type", "weather_code", "stage"])
    mode_b_idx = mode_b.set_index(["building_type", "weather_code", "stage"])

    def get_eui(bt: str, wc: str, stage: str, col: str = "net_site_eui") -> float:
        try:
            return energy_idx.loc[(bt, wc, stage), col]
        except KeyError:
            return np.nan

    def get_carbon_b(bt: str, wc: str, stage: str) -> float:
        try:
            return mode_b_idx.loc[(bt, wc, stage), "carbon_intensity_kgm2"]
        except KeyError:
            return np.nan

    rows = []
    for bt in BUILDING_TYPES:
        for wc in WEATHER_CODES:
            bl_eui = get_eui(bt, wc, "baseline", "total_site_eui")
            ecm_eui = get_eui(bt, wc, "ecm", "total_site_eui")
            pv_net_eui = get_eui(bt, wc, "pv", "net_site_eui")
            bcrc_energy = (
                (bl_eui - max(0, pv_net_eui)) / bl_eui * 100 if bl_eui else np.nan
            )

            bl_cb = get_carbon_b(bt, wc, "baseline")
            ecm_cb = get_carbon_b(bt, wc, "ecm")
            pv_cb = get_carbon_b(bt, wc, "pv")
            bcrc_cb = (bl_cb - pv_cb) / bl_cb * 100 if bl_cb else np.nan

            # Mode A carbon intensity (MidCase, 2025)
            # PV stage
            a_pv_row = df_carbon_a[
                (df_carbon_a["building_type"] == bt)
                & (df_carbon_a["weather_code"] == wc)
                & (df_carbon_a["stage"] == "pv")
                & (df_carbon_a["cambium_scenario"] == "MidCase")
                & (df_carbon_a["cambium_year"] == 2025)
            ]
            pv_ca = (
                a_pv_row.iloc[0]["carbon_intensity_kgm2"]
                if not a_pv_row.empty
                else np.nan
            )

            # Baseline stage (Mode A)
            a_bl_row = df_carbon_a[
                (df_carbon_a["building_type"] == bt)
                & (df_carbon_a["weather_code"] == wc)
                & (df_carbon_a["stage"] == "baseline")
                & (df_carbon_a["cambium_scenario"] == "MidCase")
                & (df_carbon_a["cambium_year"] == 2025)
            ]
            bl_ca = (
                a_bl_row.iloc[0]["carbon_intensity_kgm2"]
                if not a_bl_row.empty
                else np.nan
            )

            # BCRC_carbon_A: Mode A baseline as denominator
            bcrc_ca = (
                (bl_ca - pv_ca) / bl_ca * 100
                if (not np.isnan(bl_ca) and bl_ca != 0 and not np.isnan(pv_ca))
                else np.nan
            )

            # Mode C: find R where carbon_total_intensity ≈ 0
            mc = mode_c[
                (mode_c["building_type"] == bt) & (mode_c["weather_code"] == wc)
            ].sort_values("R")
            r_neutrality = np.nan
            if not mc.empty:
                for j in range(len(mc) - 1):
                    c0 = mc.iloc[j]["carbon_total_intensity_kgm2"]
                    c1 = mc.iloc[j + 1]["carbon_total_intensity_kgm2"]
                    r0 = mc.iloc[j]["R"]
                    r1 = mc.iloc[j + 1]["R"]
                    if c0 >= 0 and c1 <= 0 and c1 != c0:
                        r_neutrality = r0 + (0 - c0) * (r1 - r0) / (c1 - c0)
                        break
                else:
                    if mc.iloc[-1]["carbon_total_intensity_kgm2"] > 0:
                        r_neutrality = np.inf
                    else:
                        r_neutrality = mc.iloc[0]["R"]

            rows.append(
                {
                    "building_type": bt,
                    "weather_code": wc,
                    "baseline_site_eui": bl_eui,
                    "ecm_site_eui": ecm_eui,
                    "pv_net_site_eui": pv_net_eui,
                    "bcrc_energy_pct": bcrc_energy,
                    "baseline_carbon_mode_b": bl_cb,
                    "ecm_carbon_mode_b": ecm_cb,
                    "pv_carbon_mode_b": pv_cb,
                    "bcrc_carbon_mode_b_pct": bcrc_cb,
                    "baseline_carbon_mode_a_mid2025": bl_ca,
                    "pv_carbon_mode_a_mid2025": pv_ca,
                    "bcrc_carbon_mode_a_pct": bcrc_ca,
                    "r_neutrality_mode_c": r_neutrality,
                }
            )

    df = pd.DataFrame(rows)
    logger.info(f"  → {len(df)} rows")
    return df


# ── Validation ───────────────────────────────────────────────────────────────


def validate(
    df_energy: pd.DataFrame,
    df_elec: pd.DataFrame,
    df_hourly: pd.DataFrame,
    df_carbon_bc: pd.DataFrame,
    df_bcrc: pd.DataFrame,
) -> bool:
    """Run validation checks and report results."""
    logger.info("Running validation checks...")
    ok = True

    # 1. Row counts
    assert len(df_energy) == 90, f"CSV 1 should have 90 rows, got {len(df_energy)}"
    assert len(df_elec) == 90, f"CSV 3 should have 90 rows, got {len(df_elec)}"

    # 2. Hourly sum vs annual for all buildings
    for bt in BUILDING_TYPES:
        for wc in WEATHER_CODES:
            h = df_hourly[
                (df_hourly["building_type"] == bt) & (df_hourly["weather_code"] == wc)
            ]
            a = df_elec[
                (df_elec["building_type"] == bt)
                & (df_elec["weather_code"] == wc)
                & (df_elec["stage"] == "pv")
            ]
            if h.empty or a.empty:
                continue
            hourly_purchased = h["purchased_kw"].sum()
            annual_purchased = a.iloc[0]["purchased_electricity_kwh"]
            if annual_purchased > 0:
                rel_err = (
                    abs(hourly_purchased - annual_purchased) / annual_purchased * 100
                )
                if rel_err > 1.0:
                    logger.warning(
                        f"  {bt}/{wc}: hourly purchased ({hourly_purchased:.0f}) vs "
                        f"annual ({annual_purchased:.0f}): {rel_err:.2f}% error"
                    )
                    ok = False

            hourly_exported = h["exported_kw"].sum()
            annual_exported = a.iloc[0]["exported_electricity_kwh"]
            if not np.isnan(annual_exported) and annual_exported > 100:
                rel_err_exp = (
                    abs(hourly_exported - annual_exported) / annual_exported * 100
                )
                if rel_err_exp > 1.0:
                    logger.warning(
                        f"  {bt}/{wc}: hourly exported ({hourly_exported:.0f}) vs "
                        f"annual ({annual_exported:.0f}): {rel_err_exp:.2f}% error"
                    )
                    ok = False

    logger.info("  ✓ Hourly/annual consistency check complete")

    # 3. Mode B manual check (OfficeMedium TMY baseline)
    #    purchased=327614.476, exported=221.389, gas=112636.63
    mode_b = df_carbon_bc[df_carbon_bc["mode"] == "mode_b"]
    mb_row = mode_b[
        (mode_b["building_type"] == "OfficeMedium")
        & (mode_b["weather_code"] == "TMY")
        & (mode_b["stage"] == "baseline")
    ]
    if not mb_row.empty:
        expected_elec_carbon = 327614.476 * EF_ELEC_AVG
        expected_gas_carbon = 112636.63 * EF_GAS
        actual_net = mb_row.iloc[0]["carbon_net_kg"]
        expected_net = (
            expected_elec_carbon - 221.389 * EF_ELEC_MARGINAL + expected_gas_carbon
        )
        rel = abs(actual_net - expected_net) / expected_net * 100
        logger.info(f"  Mode B check OfficeMedium/TMY/baseline: {rel:.3f}% diff")

    # 4. BCRC energy check (OfficeMedium TMY)
    bcrc_row = df_bcrc[
        (df_bcrc["building_type"] == "OfficeMedium")
        & (df_bcrc["weather_code"] == "TMY")
    ]
    if not bcrc_row.empty:
        bcrc_val = bcrc_row.iloc[0]["bcrc_energy_pct"]
        logger.info(f"  BCRC energy OfficeMedium/TMY: {bcrc_val:.1f}%")

    if ok:
        logger.info("All validation checks passed!")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {PAPER_DIR}")

    # CSV 1: Energy summary
    df_energy = prepare_energy_summary()
    df_energy.to_csv(PAPER_DIR / "01_energy_summary.csv", index=False)

    # CSV 2: Surrogate evaluation
    df_eval = prepare_surrogate_evaluation()

    # CSV 3: Electricity balance
    df_elec = prepare_electricity_balance()
    df_elec.to_csv(PAPER_DIR / "03_electricity_balance.csv", index=False)

    # Update surrogate pred_error_pct with actual data
    df_eval = update_surrogate_pred_error(df_eval, df_energy)
    df_eval.to_csv(PAPER_DIR / "02_surrogate_evaluation.csv", index=False)

    # CSV 4: Hourly power
    df_hourly = prepare_hourly_power()
    df_hourly.to_csv(PAPER_DIR / "04_hourly_power.csv", index=False)

    # CSV 5: Carbon Mode B + C
    df_carbon_bc = prepare_carbon_mode_bc(df_elec, df_energy)
    df_carbon_bc.to_csv(PAPER_DIR / "05_carbon_mode_bc.csv", index=False)

    # CSV 6: Carbon Mode A (Cambium)
    df_carbon_a = prepare_carbon_mode_a(df_hourly, df_elec, df_energy)
    df_carbon_a.to_csv(PAPER_DIR / "06_carbon_mode_a.csv", index=False)

    # CSV 7: BCRC Summary
    df_bcrc = prepare_bcrc_summary(df_energy, df_carbon_bc, df_carbon_a)
    df_bcrc.to_csv(PAPER_DIR / "07_bcrc_summary.csv", index=False)

    # Validate
    validate(df_energy, df_elec, df_hourly, df_carbon_bc, df_bcrc)

    logger.info(f"Done! All CSV files saved to {PAPER_DIR}")


if __name__ == "__main__":
    main()
