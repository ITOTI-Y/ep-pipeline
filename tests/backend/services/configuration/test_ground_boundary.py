from pathlib import Path

import pytest

from backend.services.configuration.ground_boundary import (
    annual_mean_dry_bulb,
    bottom_edge,
    exposed_perimeter,
    floor_shared_length,
    shared_length,
)

FLOOR = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 6.0, 0.0), (0.0, 6.0, 0.0)]


def _wall(x0: float, y0: float, x1: float, y1: float, z0: float = 0.0):
    return [(x0, y0, z0 + 3.0), (x0, y0, z0), (x1, y1, z0), (x1, y1, z0 + 3.0)]


def test_shared_length_counts_partial_overlap() -> None:
    edge = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert shared_length(edge, ((4.0, 0.0, 0.0), (14.0, 0.0, 0.0))) == pytest.approx(
        6.0
    )


def test_shared_length_ignores_parallel_offset_segment() -> None:
    edge = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert shared_length(edge, ((0.0, 0.2, 0.0), (10.0, 0.2, 0.0))) == 0.0


def test_bottom_edge_returns_lowest_segment() -> None:
    edge = bottom_edge(_wall(10.0, 0.0, 10.0, 6.0, z0=-5.0))
    assert {edge[0][:2], edge[1][:2]} == {(10.0, 0.0), (10.0, 6.0)}
    assert edge[0][2] == -5.0


def test_exposed_perimeter_sums_walls_at_floor_height() -> None:
    walls = [
        bottom_edge(_wall(0.0, 0.0, 10.0, 0.0)),
        bottom_edge(_wall(10.0, 0.0, 10.0, 6.0)),
        bottom_edge(_wall(0.0, 6.0, 10.0, 6.0, z0=3.0)),
    ]
    assert exposed_perimeter(FLOOR, walls) == pytest.approx(16.0)


def test_floor_shared_length_rejects_wall_off_the_outline() -> None:
    inner = bottom_edge(_wall(5.0, 0.0, 5.0, 6.0))
    assert floor_shared_length(FLOOR, inner) == 0.0


def test_annual_mean_dry_bulb_reads_epw_records(tmp_path: Path) -> None:
    header = ["HEADER"] * 8
    records = [f"2001,1,1,{hour},0,?,{temp}" for hour, temp in enumerate((1.0, 3.0))]
    epw = tmp_path / "site.epw"
    epw.write_text("\n".join([*header, *records]) + "\n")
    assert annual_mean_dry_bulb(epw) == pytest.approx(2.0)
