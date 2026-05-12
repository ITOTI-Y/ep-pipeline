from backend.citys._share import VizFileName
from backend.viz.generator import ChartGenerator
from backend.viz.style import FigureWidth

VIZ_FILE_NAME = VizFileName()

_chart_generator = ChartGenerator()


def station_distribution() -> None:
    fig, ax = _chart_generator.create_figure(
        width=FigureWidth.DOUBLE_COLUMN,
    )
