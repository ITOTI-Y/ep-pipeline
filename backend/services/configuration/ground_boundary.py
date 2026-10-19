import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

type Point = tuple[float, float, float]
type Edge = tuple[Point, Point]

COLLINEAR_TOLERANCE_M: Final = 0.05
EPW_HEADER_LINES: Final = 8
EPW_DRY_BULB_COLUMN: Final = 6


def horizontal_length(edge: Edge) -> float:
    (ax, ay, _), (bx, by, _) = edge
    return math.hypot(bx - ax, by - ay)


def shared_length(edge: Edge, other: Edge) -> float:
    """Return the plan-view length of ``other`` lying on ``edge``.

    Both segments are projected onto the horizontal plane. ``other`` counts
    only when both of its endpoints lie within ``COLLINEAR_TOLERANCE_M`` of the
    line through ``edge``.
    """
    (ax, ay, _), (bx, by, _) = edge
    length = math.hypot(bx - ax, by - ay)
    if length == 0.0:
        return 0.0
    ux, uy = (bx - ax) / length, (by - ay) / length
    offsets = []
    for px, py, _ in other:
        dx, dy = px - ax, py - ay
        if abs(dx * uy - dy * ux) > COLLINEAR_TOLERANCE_M:
            return 0.0
        offsets.append(dx * ux + dy * uy)
    start = max(0.0, min(offsets))
    end = min(length, max(offsets))
    return max(0.0, end - start)


def polygon_edges(vertices: Sequence[Point]) -> list[Edge]:
    return [
        (vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))
    ]


def bottom_edge(vertices: Sequence[Point]) -> Edge:
    """Return the longest horizontal segment at the lowest height of a wall."""
    z_min = min(z for _, _, z in vertices)
    low = [p for p in vertices if abs(p[2] - z_min) < 1e-3]
    if len(low) < 2:
        raise ValueError(f"Wall has fewer than two vertices at z={z_min}")
    return max(
        ((a, b) for i, a in enumerate(low) for b in low[i + 1 :]),
        key=horizontal_length,
    )


def floor_shared_length(floor: Sequence[Point], edge: Edge) -> float:
    return sum(shared_length(floor_edge, edge) for floor_edge in polygon_edges(floor))


def exposed_perimeter(floor: Sequence[Point], wall_bottoms: Sequence[Edge]) -> float:
    """Return the floor perimeter covered by wall bottoms at the floor height."""
    z = floor[0][2]
    return sum(
        floor_shared_length(floor, wall)
        for wall in wall_bottoms
        if abs(wall[0][2] - z) < COLLINEAR_TOLERANCE_M
    )


def annual_mean_dry_bulb(epw_path: Path) -> float:
    lines = epw_path.read_text(errors="ignore").splitlines()[EPW_HEADER_LINES:]
    values = [float(line.split(",")[EPW_DRY_BULB_COLUMN]) for line in lines if line]
    if not values:
        raise ValueError(f"No weather records in {epw_path}")
    return sum(values) / len(values)
