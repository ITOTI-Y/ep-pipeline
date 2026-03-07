import functools
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pickle import load
from typing import Any, Literal

import numpy as np
import pandas as pd
import ultraplot as uplt

from backend.models import SimulationResult
from backend.utils.config import ConfigManager
from backend.visualization.journal_style import (
    BUILDING_AND_ENVIRONMENT_STYLE,
    FigureWidth,
    ImageType,
    JournalStyle,
)

WEATHER_ORDER = ["TMY", "SSP126", "SSP245", "SSP370", "SSP434", "SSP585"]
BUILDING_ORDER = [
    "OfficeLarge",
    "OfficeMedium",
    "MultiFamilyResidential",
    "SingleFamilyResidential",
    "ApartmentHighRise",
]

BUILDING_NAME = {
    "OfficeLarge": "Large Office",
    "OfficeMedium": "Medium Office",
    "MultiFamilyResidential": "Multi-Family",
    "SingleFamilyResidential": "Single-Family",
    "ApartmentHighRise": "High-Rise Apt.",
}

BUILDING_COLOR = {
    "OfficeLarge": BUILDING_AND_ENVIRONMENT_STYLE.get_color(0),
    "OfficeMedium": BUILDING_AND_ENVIRONMENT_STYLE.get_color(1),
    "MultiFamilyResidential": BUILDING_AND_ENVIRONMENT_STYLE.get_color(2),
    "SingleFamilyResidential": BUILDING_AND_ENVIRONMENT_STYLE.get_color(3),
    "ApartmentHighRise": BUILDING_AND_ENVIRONMENT_STYLE.get_color(4),
}

GAS_BUILDING_TYPE = ["OfficeLarge", "OfficeMedium", "ApartmentHighRise"]

SEASON_TAG = ["Summer (Jul 14-16)", "Winter (Jan 14-16)"]

SUMMER_DAYS = [14, 15, 16]
WINTER_DAYS = [14, 15, 16]

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_HOURS = np.concatenate([[0], np.cumsum([d * 24 for d in DAYS_IN_MONTH])])
MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
MONTH_CENTERS = (MONTH_HOURS[:-1] + MONTH_HOURS[1:]) / 2


CAMBIUM_SCENARIOS = {
    "MidCase": ("-", "Mid-case"),
    "LowRECost": ("--", "Low RE Cost"),
    "HighRECost_LowNGPrice": (":", "High RE+Low Gas"),
}

STAGE_LABEL = {"baseline": "Baseline", "ecm": "Post-ECM", "pv": "Post-PV"}

STAGE_CFG = {
    "baseline": "total_site_eui",
    "ecm": "total_site_eui",
    "pv": "net_site_eui",
}


@dataclass
class Prefix:
    baseline: str = "B"
    pv: str = "P"
    optimization: str = "O"


def hour_of_year(dt: datetime) -> int:
    year_start = datetime(2017, 1, 1)
    delta = dt - year_start
    return int(delta.total_seconds() // 3600)


class ChartGenerator:
    """Chart generator for PV simulation results visualization."""

    def __init__(
        self,
        config: ConfigManager,
        style: JournalStyle | None = None,
    ):
        self.config = config
        self.paths = config.paths
        self.csv_dir = config.paths.csv_dir
        self.output_dir = config.paths.visualization_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style or BUILDING_AND_ENVIRONMENT_STYLE
        uplt.rc.update(self.style.get_rc_params())
        self.pv_results = self._load_results(self.paths.pv_dir)
        self.baseline_results = self._load_results(self.paths.baseline_dir)
        self.optimization_results = self._load_results(self.paths.optimization_dir)

    def _load_results(self, directory: Path) -> dict[str, dict[str, SimulationResult]]:
        results: dict[str, dict[str, SimulationResult]] = defaultdict(dict)
        for pkl_file in directory.glob("**/result.pkl"):
            with open(pkl_file, "rb") as f:
                data = load(f)
                results[data.building_type][data.sql_path.parent.name] = data
        return results

    @functools.cached_property
    def _hourly_power(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_dir / "04_hourly_power.csv")

    @functools.cached_property
    def _carbon_mode_bc(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_dir / "05_carbon_mode_bc.csv")

    @functools.cached_property
    def _carbon_mode_a(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_dir / "06_carbon_mode_a.csv")

    @functools.cached_property
    def _bcrc_summary(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_dir / "07_bcrc_summary.csv")

    @functools.cached_property
    def _energy_summary(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_dir / "01_energy_summary.csv")

    def create_figure(
        self,
        width: FigureWidth = FigureWidth.SINGLE_COLUMN,
        aspect_ratio: float = 0.3,
        ncols: int = 1,
        nrows: int = 1,
        **kwargs,
    ) -> tuple[uplt.Figure, Any]:
        """Create figure with journal-compliant dimensions.

        Args:
            width: Total figure width (not per-subplot).
            aspect_ratio: Height/width ratio for each subplot.
            ncols: Number of columns.
            nrows: Number of rows.
        """
        subplot_width = width.value / ncols
        subplot_height = subplot_width * aspect_ratio
        return uplt.subplots(
            ncols=ncols,
            nrows=nrows,
            refwidth=subplot_width,
            refheight=subplot_height,
            **kwargs,
        )

    def save(
        self,
        fig: uplt.Figure,
        name: str,
        building_type: str | None = None,
        image_type: ImageType = ImageType.LINE_ART,
        fmt: str | None = None,
    ) -> Path:
        """Save figure with journal-compliant format and DPI."""
        image_dir = self.output_dir / (building_type if building_type else "")
        image_dir.mkdir(parents=True, exist_ok=True)
        output_format = fmt or self.style.default_format
        path = image_dir / f"{name}.{output_format}"
        fig.save(path, dpi=image_type.value)
        return path

    def storage_soc(self, weather_code: str = "TMY", smooth_window: int = 7) -> None:
        df = self._hourly_power
        df = df[df["weather_code"] == weather_code]
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.15,
            ncols=1,
            nrows=len(BUILDING_ORDER) - 1,
        )
        idx = 0
        for building_type in BUILDING_ORDER:
            ax = axs[idx]
            sub = df[df["building_type"] == building_type].sort_values(
                ["month", "day", "hour"]
            )
            capacity = self.config.storage.capacity[building_type]
            if capacity == 0:
                continue

            soc_pct = sub["storage_soc_kwh"].values / capacity * 100

            soc_daily = soc_pct.reshape(365, 24)
            day_min = soc_daily.min(axis=1)
            day_max = soc_daily.max(axis=1)
            day_mean = soc_daily.mean(axis=1)
            day_x = np.arange(365) * 24 + 12

            day_mean_smooth = (
                pd.Series(day_mean)
                .rolling(smooth_window, center=True, min_periods=1)
                .mean()
                .values
            )
            ax.area(
                day_x,
                day_min,
                day_max,
                alpha=0.30,
                linewidth=0,
                color=BUILDING_COLOR[building_type],
                label="Storage State of Charge",
            )
            ax.plot(
                day_x,
                day_mean_smooth,
                linewidth=self.style.line_width,
                color=BUILDING_COLOR[building_type],
                label="Storage State of Charge",
            )
            ax.text(
                0.98,
                0.90,
                f"{BUILDING_NAME[building_type]} - {capacity} kWh",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=self.style.font_size,
            )

            idx += 1

        axs.format(
            abc="a",
            abcloc="ul",
            xlim=(0, 8760),
            xlocator=MONTH_CENTERS.tolist(),
            xticklabels=MONTH_LABELS,
            ylabel="SOC (%)",
            xlabel="Month",
            suptitle=f"Storage SOC - {weather_code}",
        )

        self.save(fig, f"Fig11. {Prefix.pv}-Storage SOC - {weather_code}")

    def _load_typical_day_data(
        self,
        df: pd.DataFrame,
        building_type: str,
        month: int,
        days: list[int],
        weather_code: str = "TMY",
    ) -> pd.DataFrame:
        mask = (
            (df["building_type"] == building_type)
            & (df["month"] == month)
            & (df["day"].isin(days))
            & (df["weather_code"] == weather_code)
        )
        sub = df[mask].copy().sort_values(["day", "hour"]).reset_index(drop=True)

        sub["hour_cont"] = np.arange(1, len(sub) + 1)

        for col in [
            "demand_kw",
            "pv_generation_kw",
            "purchased_kw",
            "exported_kw",
            "storage_charge_kw",
            "storage_discharge_kw",
        ]:
            sub[col] = sub[col].clip(lower=0)

        sub["pv_self_kw"] = (
            sub["pv_generation_kw"] - sub["exported_kw"] - sub["storage_charge_kw"]
        ).clip(lower=0)

        return sub

    def _typical_single_plot(self, ax: uplt.PlotAxes, data: pd.DataFrame) -> None:
        h = data["hour_cont"].values
        pv_s = data["pv_self_kw"].values
        bd = data["storage_discharge_kw"].values
        gp = data["purchased_kw"].values
        bc = data["storage_charge_kw"].values
        ge = data["exported_kw"].values
        dem = data["demand_kw"].values

        ax.area(h, 0, pv_s, alpha=0.6, linewidth=0, label="PV Self-Consumption")
        ax.area(h, pv_s, pv_s + bd, alpha=0.6, linewidth=0, label="Battery Discharge")
        ax.area(
            h, pv_s + bd, pv_s + bd + gp, alpha=0.6, linewidth=0, label="Grid Purchase"
        )

        ax.area(h, 0, -bc, alpha=0.6, linewidth=0, label="Battery Charge")
        ax.area(h, -bc, -bc - ge, alpha=0.6, linewidth=0, label="Grid Export")

        ax.plot(
            h, dem, linewidth=self.style.line_width_thick, color="black", label="Demand"
        )
        ax.axhline(0, color="grey", linewidth=self.style.line_width / 2, zorder=1)

        for boundary in np.arange(0, max(h), 24):
            ax.axvline(
                boundary,
                color="grey",
                linewidth=self.style.line_width / 2,
                linestyle=":",
                alpha=0.6,
                zorder=1,
            )
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 1.2)

    def typical_day_storage_soc(self, weather_code: str = "TMY") -> None:
        df = self._hourly_power
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.25,
            ncols=2,
            nrows=len(BUILDING_ORDER),
        )

        for col_idx, season in enumerate(SEASON_TAG):
            axs[0, col_idx].set_title(
                season, fontsize=self.style.font_size_title, fontweight="bold"
            )

        label = []

        for row_idx, building_type in enumerate(BUILDING_ORDER):
            label.extend(
                [
                    chr(ord("a") + row_idx)
                    + f"{col_idx + 1} {BUILDING_NAME[building_type] if col_idx == 0 else ''}"
                    for col_idx in range(2)
                ]
            )
            summer = self._load_typical_day_data(
                df, building_type, month=7, days=SUMMER_DAYS, weather_code=weather_code
            )
            winter = self._load_typical_day_data(
                df, building_type, month=1, days=WINTER_DAYS, weather_code=weather_code
            )

            ax_s = axs[row_idx, 0]
            ax_w = axs[row_idx, 1]

            self._typical_single_plot(ax_s, summer)
            self._typical_single_plot(ax_w, winter)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="b", ncols=6, fontsize=self.style.font_size)

        axs.format(
            abc=label,
            abcloc="ul",
            xlim=(0, max(summer["hour_cont"].max(), winter["hour_cont"].max())),
            xticks=np.arange(
                0, max(summer["hour_cont"].max(), winter["hour_cont"].max()) + 1, 12
            ),
            suptitle=f"Typical Day Storage SOC - {weather_code}",
        )

        axs.set_xlabel("Hour", fontsize=self.style.font_size_title)
        axs.set_ylabel("Power (kW)", fontsize=self.style.font_size_title)

        self.save(fig, f"Fig10. {Prefix.pv}-Typical Day Storage SOC - {weather_code}")

    def baseline_eui_heatmap(self) -> None:
        df = self._energy_summary
        df = df[df["stage"] == "baseline"]
        pivot = df.pivot(
            index="building_type", columns="weather_code", values="total_site_eui"
        ).loc[BUILDING_ORDER, WEATHER_ORDER]

        pivot.index = [BUILDING_NAME[i] for i in pivot.index]

        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.5,
        )
        ax.heatmap(
            pivot,
            cmap="YlOrRd",
            colorbar="r",
            labels=True,
            precision=1,
        )
        ax.format(
            aspect="auto",
            xlabel="",
            title="Baseline Site EUI (kWh m$^{-2}$ yr$^{-1}$)",
            titleweight="bold",
            titleloc="l",
            titlesize=self.style.font_size_title,
            labelsize=self.style.font_size_title,
        )
        self.save(fig, f"Fig05. {Prefix.baseline}-Baseline EUI Heatmap")

    def ecm_improvement_heatmap(self) -> None:
        df = self._energy_summary
        df = df[df["stage"] == "ecm"]
        pivot = df.pivot(
            index="building_type", columns="weather_code", values="ecm_improvement_pct"
        ).loc[BUILDING_ORDER, WEATHER_ORDER]

        pivot.index = [BUILDING_NAME[i] for i in pivot.index]

        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.5,
        )
        ax.heatmap(
            pivot,
            cmap="Blues",
            colorbar="r",
            labels=True,
            formatter="simple",
            precision=1,
            formatter_kw={"suffix": "%"},
        )
        ax.format(
            aspect="auto",
            xlabel="",
            title="Source EUI Improvement by ECM Optimization (%)",
            titleweight="bold",
            titleloc="l",
            titlesize=self.style.font_size_title,
            labelsize=self.style.font_size_title,
        )
        self.save(fig, f"Fig06. {Prefix.optimization}-ECM Improvement Heatmap")

    def optimal_improvement_violin(self) -> None:
        base = np.arange(len(BUILDING_ORDER))
        offset = {"baseline": -0.25, "ecm": 0.0, "pv": 0.25}
        df = self._energy_summary
        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.5,
            ncols=1,
        )
        width = 0.2
        for i, (stage, col) in enumerate(STAGE_CFG.items()):
            pos = base + offset[stage]
            sub = df[df["stage"] == stage]
            data = pd.concat(
                [
                    sub.loc[sub["building_type"] == building_type, col].reset_index(
                        drop=True
                    )
                    for building_type in BUILDING_ORDER
                ],
                axis=1,
            )
            bodies = ax.violinplot(
                pos,
                data,
                widths=width,
                edgecolor="white",
                showmedians=False,
                showmeans=False,
                showextrema=False,
                bw_method=0.3,
                labels=[STAGE_LABEL[stage]],
            )
            color = self.style.get_color(i)
            for body in bodies:
                body.set_facecolor(color)
                body.set_alpha(0.7)

            for j, _ in enumerate(BUILDING_ORDER):
                y = data.iloc[:, j].dropna()
                jitter = np.random.uniform(-width * 0.3, width * 0.3, size=len(y))
                ax.scatter(
                    pos[j] + jitter,
                    y,
                    s=20,
                    color=color,
                    alpha=0.8,
                    edgecolor="white",
                    zorder=3,
                    label="_nolegend_",
                )
        for i in range(len(BUILDING_ORDER)):
            ax.axvline(
                i + 0.5,
                color="grey",
                linestyle=":",
                alpha=0.8,
            )
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            alpha=0.8,
            linewidth=self.style.line_width,
            zorder=1,
        )
        xticks = np.arange(len(BUILDING_ORDER))
        xticklabels = [BUILDING_NAME[building_type] for building_type in BUILDING_ORDER]
        ax.format(
            xticks=xticks,
            xticklabels=xticklabels,
            xlim=(-0.5, len(BUILDING_ORDER) - 0.5),
            title="Optimization Potential Distribution Across Climate Scenarios",
            titleweight="bold",
            titleloc="l",
            titlesize=self.style.font_size_title,
            labelsize=self.style.font_size_title,
            xlabel="",
            ylabel="Site EUI (kWh m$^{-2}$ yr$^{-1}$)",
        )
        ax.legend(loc="upper right", ncols=3, frameon=False)

        self.save(fig, f"Fig09. {Prefix.pv}-Optimal Improvement Violin")

    def neutrality_timeline(self, stage: Literal["pv", "baseline"] = "pv") -> None:
        df_a = self._carbon_mode_a
        df_a = df_a[df_a["stage"] == stage]

        df_mid = df_a[df_a["cambium_scenario"] == "MidCase"].copy()
        df_avg = (
            df_mid.groupby(["building_type", "cambium_year"])
            .agg(carbon=("carbon_intensity_kgm2", "mean"))
            .reset_index()
        )

        df_all = (
            df_a.groupby(["building_type", "cambium_year"])
            .agg(
                carbon_min=("carbon_intensity_kgm2", "min"),
                carbon_max=("carbon_intensity_kgm2", "max"),
            )
            .reset_index()
        )

        df_bc = self._carbon_mode_bc
        df_c_r1 = (
            df_bc[
                (df_bc["mode"] == "mode_c")
                & (df_bc["stage"] == "pv")
                & (df_bc["R"] == 1.0)
            ]
            .groupby("building_type")["carbon_gas_intensity_kgm2"]
            .mean()
        )

        fig, ax = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.55,
        )

        ymax = df_all["carbon_max"].max() * 1.15

        for building_type in BUILDING_ORDER:
            color = BUILDING_COLOR[building_type]
            sub = df_avg[df_avg["building_type"] == building_type].sort_values(
                "cambium_year"
            )
            sub_all = df_all[df_all["building_type"] == building_type].sort_values(
                "cambium_year"
            )

            ax.area(
                sub_all["cambium_year"],
                sub_all["carbon_min"],
                sub_all["carbon_max"],
                color=color,
                alpha=0.12,
                label="_nolegend_",
            )

            ax.plot(
                sub["cambium_year"],
                sub["carbon"],
                color=color,
                linewidth=self.style.line_width,
                marker="o",
                markersize=5,
                label=BUILDING_NAME[building_type],
                zorder=5,
            )

            last = sub.iloc[-1]
            ax.text(
                last["cambium_year"] + 0.5,
                last["carbon"],
                f"{last['carbon']:.1f}",
                ha="left",
                va="center",
                fontsize=self.style.font_size_small,
                color=color,
            )

        for building_type in GAS_BUILDING_TYPE:
            color = BUILDING_COLOR[building_type]
            gas_value = df_c_r1[building_type]
            ax.axhline(
                gas_value,
                color=color,
                linestyle="--",
                linewidth=self.style.line_width,
                alpha=0.4,
            )
            ax.text(
                2051.2,
                gas_value + 0.2,
                f"Gas: {gas_value:.1f}",
                fontsize=self.style.font_size_small,
                color=color,
                alpha=0.8,
                ha="left",
                va="bottom",
            )

        ax.axhline(0, color="black", linewidth=self.style.line_width)
        ax.text(
            2024.5,
            0.5,
            "Carbon Neutral",
            fontsize=self.style.font_size,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="green",
        )

        policy = {
            2030: ("Illinois RPS\n40%", "grey"),
            2040: ("Illinois RPS\n50%", "grey"),
            2045: ("CEJA\n100% Clean", self.style.get_color(4)),
        }
        for year, (label, color) in policy.items():
            ax.axvline(
                year,
                color=color,
                linestyle=":",
                linewidth=self.style.line_width,
                alpha=0.6,
            )
            ax.text(
                year,
                ymax * 0.97,
                label,
                ha="center",
                va="top",
                fontsize=self.style.font_size_small,
                color=color,
                fontweight="bold" if year == 2045 else "normal",
            )

        ax.format(
            xlabel="Year",
            ylabel="Carbon Intensity (kg CO₂e/m²/yr)",
            xlim=(2024, 2053),
            ylim=(-10, ymax),
            title="Carbon Neutrality Pathway Timeline (Mode A, MidCase with Scenario Bands)",
            titleweight="bold",
            titlesize=self.style.font_size_title,
            labelsize=self.style.font_size_title,
        )
        ax.legend(loc="ur", ncol=1)

        self.save(
            fig,
            f"Fig15. {Prefix.pv}-Carbon Neutrality Pathway Timeline",
            building_type=None,
        )

    def waterfall(self) -> None:
        df = self._bcrc_summary
        means = (
            df.groupby("building_type")[
                [
                    "baseline_site_eui",
                    "ecm_site_eui",
                    "pv_net_site_eui",
                    "bcrc_energy_pct",
                    "bcrc_carbon_mode_b_pct",
                    "bcrc_carbon_mode_a_pct",
                ]
            ]
            .mean()
            .reindex(BUILDING_ORDER)
        )

        baseline = means["baseline_site_eui"].values
        ecm = means["ecm_site_eui"].values
        pv_net = means["pv_net_site_eui"].values
        ecm_reduction = baseline - ecm
        pv_reduction = ecm - pv_net
        net_eui = pv_net.copy()

        bcrc_energy = means["bcrc_energy_pct"].values
        bcrc_carbon_b = means["bcrc_carbon_mode_b_pct"].values
        bcrc_carbon_a = means["bcrc_carbon_mode_a_pct"].values

        x = np.arange(len(BUILDING_ORDER))
        bar_width = 0.55
        bar_label_offset = 0.03
        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=0.3,
            ncols=1,
            nrows=2,
            sharey=False,
        )

        axs.format(
            abc="a",
            abcloc="ul",
        )

        ax1 = axs[0]
        c_net, c_pv, c_ecm = (
            "#BDBDBD",
            self.style.get_color(2),
            self.style.get_color(0),
        )

        ax1.axhline(0, color="black", linewidth=self.style.line_width / 2)

        ax1.bar(
            x,
            net_eui,
            width=bar_width,
            color=c_net,
            bottom=np.zeros(len(BUILDING_ORDER)),
            edgecolor="white",
            linewidth=self.style.line_width / 2,
            label="Net site EUI",
        )
        ax1.bar(
            x,
            pv_reduction,
            width=bar_width,
            color=c_pv,
            bottom=net_eui,
            edgecolor="white",
            linewidth=self.style.line_width / 2,
            label="PV reduction",
            alpha=0.85,
        )
        ax1.bar(
            x,
            ecm_reduction,
            width=bar_width,
            color=c_ecm,
            bottom=net_eui + pv_reduction,
            edgecolor="white",
            linewidth=self.style.line_width / 2,
            label="ECM reduction",
            alpha=0.85,
        )

        for i in range(len(x)):
            bsl, er, pr, nv = baseline[i], ecm_reduction[i], pv_reduction[i], net_eui[i]

            ax1.text(
                x[i],
                bsl + 2.5,
                f"{bsl:.1f}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=self.style.font_size,
                fontweight="bold",
            )
            ax1.text(
                x[i] + bar_width / 2 + bar_label_offset,
                nv + pr + er / 2,
                f"- {er:.1f}",
                fontsize=self.style.font_size_small,
                color=c_ecm,
                ha="left",
                va="center",
                fontweight="bold",
            )
            ax1.text(
                x[i] + bar_width / 2 + bar_label_offset,
                nv + pr / 2,
                f"- {pr:.1f}",
                fontsize=self.style.font_size_small,
                color=c_pv,
                ha="left",
                va="center",
                fontweight="bold",
            )
            if nv > 20:
                ax1.text(
                    x[i],
                    nv / 2,
                    f"{nv:.1f}",
                    fontsize=self.style.font_size_small,
                    color="#555555",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
            else:
                ax1.text(
                    x[i],
                    nv - 7 if nv < 2 else nv / 2,
                    f"{nv:.1f}",
                    fontsize=self.style.font_size_small,
                    color="#555555",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
        ax1.legend(loc="ur")
        y_min, ymax = ax1.get_ylim()
        ax1.format(
            xlim=(x[0] - 0.5, x[-1] + 0.75),
            ylim=(y_min - 20 if y_min < 0 else 0, ymax + 20),
            ylabel="Site EUI (kWh/m²/yr)",
            title="Three-stage energy reduction pathway (6-scenario mean)",
        )

        ax2 = axs[1]
        gbw = 0.75 / 3
        offset = [-gbw, 0, gbw]
        c_bcrc = [
            self.style.get_color(0),
            self.style.get_color(1),
            self.style.get_color(2),
        ]
        bcrc_lbl = ["BCRC$_{energy}$", "BCRC$_{carbon,B}$", "BCRC$_{carbon,A}$"]
        bcrc_data = [bcrc_energy, bcrc_carbon_b, bcrc_carbon_a]

        for j in range(3):
            ax2.bar(
                x + offset[j],
                bcrc_data[j],
                width=gbw * 0.85,
                color=c_bcrc[j],
                edgecolor="white",
                linewidth=self.style.line_width / 2,
                label=bcrc_lbl[j],
            )
            for i_b in range(len(x)):
                v = bcrc_data[j][i_b]
                ax2.text(
                    x[i_b] + offset[j],
                    v + 4,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    color=c_bcrc[j],
                    fontsize=self.style.font_size_small,
                    fontweight="bold",
                    rotation=45,
                )
        ax2.axhline(
            100,
            color="grey",
            linewidth=self.style.line_width / 2,
            linestyle="--",
            alpha=0.8,
        )
        ax2.text(
            4.5,
            100,
            "Carbon neutral threshold",
            fontsize=self.style.font_size_small,
            color="grey",
            alpha=0.8,
            ha="right",
            va="bottom",
            style="italic",
        )
        y2_min, y2_max = ax2.get_ylim()
        ax2.format(
            xlim=(x[0] - 0.5, x[-1] + 0.75),
            ylim=(y2_min - 20 if y2_min < 0 else 0, y2_max + 65),
            xticks=x,
            xticklabels=list(BUILDING_NAME.values()),
            ylabel="BCRC (%)",
            title="Building Carbon Reduction Contribution (BCRC, 6-scenario mean)",
        )
        ax2.legend(loc="ur")

        self.save(fig, f"Fig12. {Prefix.pv}-Waterfall Chart", building_type=None)

    def carbon_three_plane(self, weather_code: str = "TMY") -> None:
        mc_bc = self._carbon_mode_bc
        ma = self._carbon_mode_a

        mode_c_tmy = mc_bc[
            (mc_bc["mode"] == "mode_c") & (mc_bc["weather_code"] == weather_code)
        ]
        mode_b_pv_tmy = mc_bc[
            (mc_bc["mode"] == "mode_b")
            & (mc_bc["weather_code"] == weather_code)
            & (mc_bc["stage"] == "pv")
        ]

        ma_pv_tmy = ma[(ma["stage"] == "pv") & (ma["weather_code"] == weather_code)]

        fig, axs = self.create_figure(
            width=FigureWidth.DOUBLE_COLUMN,
            aspect_ratio=1.0,
            ncols=3,
            nrows=1,
            sharey=False,
            sharex=False,
        )

        axs.format(
            abc="a",
            abcloc="ul",
        )

        ax1 = axs[0]
        for building_type in BUILDING_ORDER:
            for scenario, (linestyle, _) in CAMBIUM_SCENARIOS.items():
                sub = ma_pv_tmy[
                    (ma_pv_tmy["building_type"] == building_type)
                    & (ma_pv_tmy["cambium_scenario"] == scenario)
                ].sort_values("cambium_year")
                alpha = 1 if scenario == "MidCase" else 0.5
                ax1.plot(
                    sub["cambium_year"],
                    sub["carbon_intensity_kgm2"],
                    linestyle=linestyle,
                    color=BUILDING_COLOR[building_type],
                    alpha=alpha,
                    label="_nolegend_",
                )

        building_handles = []
        for b in BUILDING_ORDER:
            h = ax1.plot(
                [],
                [],
                color=BUILDING_COLOR[b],
                linestyle="-",
                linewidth=self.style.line_width_thick,
                label=BUILDING_NAME[b],
            )[0]
            building_handles.append(h)

        scenario_handles = []
        for scenario, (linestyle, scenario_name) in CAMBIUM_SCENARIOS.items():
            alpha = 1 if scenario == "MidCase" else 0.5
            h = ax1.plot(
                [],
                [],
                color="black",
                linestyle=linestyle,
                linewidth=self.style.line_width,
                alpha=alpha,
                label=scenario_name,
            )[0]
            scenario_handles.append(h)
        leg_building = ax1.legend(
            handles=building_handles,
            ncols=1,
            loc="ur",
        )
        ax1.add_artist(leg_building)
        ax1.legend(
            handles=scenario_handles,
            loc="ul",
            bbox_to_anchor=(0.1, 1),
            ncols=1,
        )
        ax1.format(
            xlim=(2024, 2051),
            ylim=(-8, 24),
            xlabel="Cambium projection year",
            ylabel="Carbon Intensity (kg CO₂e/m²/yr)",
            title=f"Mode A: Cambium hourly - {weather_code}",
        )

        ax2 = axs[1]
        for building_type in BUILDING_ORDER:
            sub = mode_c_tmy[mode_c_tmy["building_type"] == building_type].sort_values(
                "R"
            )
            ax2.plot(
                sub["R"],
                sub["carbon_total_intensity_kgm2"],
                color=BUILDING_COLOR[building_type],
                linewidth=self.style.line_width_thick,
                label=BUILDING_NAME[building_type],
            )
            if building_type in GAS_BUILDING_TYPE:
                v = sub[sub["R"] == 1.0].iloc[0]["carbon_total_intensity_kgm2"]
                ax2.scatter(
                    1.0,
                    v,
                    color=BUILDING_COLOR[building_type],
                    marker="o",
                    markersize=20,
                )
        ax2.legend(loc="ur", ncols=1)
        ax2.format(
            xlim=(0.0, 1.05),
            ylim=(-3, 54),
            xlabel="Grid decarbonization fraction R",
            ylabel="Carbon Intensity (kg CO₂e/m²/yr)",
            title=f"Mode C: Parametric C - {weather_code}",
        )
        ax2.annotate(
            "Gas residual",
            xy=(1.0, 9.6),
            xytext=(0.7, 20),
            color="#555555",
            fontsize=self.style.font_size_small,
            ha="left",
            va="bottom",
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "linewidth": self.style.line_width,
            },
        )

        ax3 = axs[2]
        bar_width = 0.8

        for i, building_type in enumerate(BUILDING_ORDER):
            a_row = ma_pv_tmy[
                (ma_pv_tmy["building_type"] == building_type)
                & (ma_pv_tmy["cambium_scenario"] == "MidCase")
                & (ma_pv_tmy["cambium_year"] == 2025)
            ]
            b_row = mode_b_pv_tmy[mode_b_pv_tmy["building_type"] == building_type]
            value_a = a_row["carbon_intensity_kgm2"].values[0]
            value_b = b_row["carbon_intensity_kgm2"].values[0]
            ratio = value_b / value_a

            ax3.bar(
                i - bar_width / 4,
                value_a,
                width=bar_width,
                color=self.style.get_color(0),
                edgecolor="white",
                linewidth=self.style.line_width / 2,
                label="Mode A (Cambium)" if i == 0 else "_nolegend_",
                alpha=0.8,
            )
            ax3.bar(
                i + bar_width / 4,
                value_b,
                width=bar_width,
                color=self.style.get_color(4),
                edgecolor="white",
                linewidth=self.style.line_width / 2,
                label="Mode B (eGRID)" if i == 0 else "_nolegend_",
                alpha=0.8,
            )
            offset = 0.5 if value_a > 0 and value_b > 0 else -3
            y_position = (
                max(value_a, value_b)
                if value_a > 0 and value_b > 0
                else min(value_a, value_b)
            )
            ax3.text(
                i,
                y_position + offset,
                f"{ratio:.1f}x",
                ha="center",
                va="bottom",
                color=self.style.get_color(-1)
                if y_position > 0
                else self.style.get_color(0),
                fontsize=self.style.font_size_small,
                fontweight="bold",
            )

        ax3.axhline(
            0,
            color="grey",
            linewidth=self.style.line_width,
            linestyle="--",
            alpha=0.5,
        )

        ax3.legend(loc="ur", ncols=1)

        ax3.format(
            xlim=(-0.5, len(BUILDING_ORDER) - 0.5),
            ylim=(-14, 62),
            xticks=np.arange(len(BUILDING_ORDER)),
            xticklabels=list(BUILDING_NAME.values()),
            xrotation=45,
            ylabel="Carbon Intensity (kg CO₂e/m²/yr)",
            title="eGRID vs Cambium",
        )

        self.save(
            fig, f"Fig14. {Prefix.pv}-Carbon Intensity Three-Plane", building_type=None
        )

    def data_to_csv(self) -> None:
        data = []
        for building_type in BUILDING_ORDER:
            for weather_code in WEATHER_ORDER:
                baseline_result = self.baseline_results[building_type][weather_code]
                optimization_result = self.optimization_results[building_type][
                    weather_code
                ]
                pv_result = self.pv_results[building_type][weather_code]
                assert baseline_result.total_source_eui is not None
                assert optimization_result.total_source_eui is not None
                assert pv_result.total_source_eui is not None
                data.append(
                    self._extract_data(
                        building_type, weather_code, "baseline", baseline_result
                    )
                )
                data.append(
                    self._extract_data(
                        building_type, weather_code, "optimization", optimization_result
                    )
                )
                data.append(
                    self._extract_data(building_type, weather_code, "pv", pv_result)
                )
        df = pd.DataFrame(data)
        df.to_csv(self.paths.visualization_dir / "Results_data.csv", index=False)

    def _extract_data(
        self,
        building_type: str,
        weather_code: str,
        data_type: str,
        data: SimulationResult,
    ) -> dict:
        result = {}
        result["data_type"] = data_type
        result["weather_code"] = weather_code
        result["building_type"] = building_type

        result["total_building_area"] = data.total_building_area
        result["net_building_area"] = data.net_building_area
        result["total_source_energy"] = data.total_source_energy
        result["total_site_energy"] = data.total_site_energy
        result["net_site_energy"] = data.net_site_energy
        result["net_source_energy"] = data.net_source_energy
        result["total_source_eui"] = data.total_source_eui
        result["net_source_eui"] = data.net_source_eui
        result["total_site_eui"] = data.total_site_eui
        result["net_site_eui"] = data.net_site_eui
        result["predicted_eui"] = data.predicted_eui
        return result

    def generate_all(self) -> None:
        self.data_to_csv()
        self.baseline_eui_heatmap()
        self.ecm_improvement_heatmap()
        self.optimal_improvement_violin()
        self.storage_soc()
        self.waterfall()
        self.carbon_three_plane()
        self.typical_day_storage_soc(weather_code="TMY")
        self.neutrality_timeline()
