from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize

from openimc.ui.main_window import CustomNavigationToolbar


def test_main_viewer_toolbar_uses_compact_logical_icon_size(qtbot):
    canvas = FigureCanvasQTAgg(Figure())
    toolbar = CustomNavigationToolbar(canvas, None)
    qtbot.addWidget(toolbar)

    assert toolbar.iconSize() == QSize(20, 20)
