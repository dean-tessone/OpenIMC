from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize

from openimc.ui.main_window import CustomNavigationToolbar


def test_main_viewer_toolbar_uses_compact_logical_icon_size(qtbot):
    canvas = FigureCanvasQTAgg(Figure())
    toolbar = CustomNavigationToolbar(canvas, None)
    qtbot.addWidget(toolbar)

    assert toolbar.iconSize() == QSize(16, 16)

    buttons = toolbar.findChildren(QtWidgets.QToolButton)
    assert buttons
    for button in buttons:
        assert button.iconSize() == QSize(16, 16)
        assert button.minimumSize() == QSize(24, 24)
        assert button.maximumSize() == QSize(24, 24)
