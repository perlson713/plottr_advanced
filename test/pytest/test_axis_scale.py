"""Tests for switching an axis between linear and logarithmic.

`Qi` against the photon number spans decades and is read on a log x axis.
This backend had no such switch at all (the pyqtgraph one has it in its
right-click menu), so the `Photon axis` node took `log10` of the values
instead -- which works, but labels the axis `log10(photon)` with ticks at
5.6, 6.6, ... rather than at the decades.
"""

import numpy as np
import pytest

from plottr.data.datadict import DataDict
from plottr.plot.base import PlotWidgetContainer
from plottr.plot.mpl.autoplot import AutoPlot

_WIDGETS = []


@pytest.fixture
def plotWidget(qtbot):
    container = PlotWidgetContainer()
    plot = AutoPlot(container)
    _WIDGETS.append((container, plot))
    return plot


def _decades():
    data = DataDict(photon=dict(), Qi=dict(axes=['photon']))
    data.validate()
    for n in (1.0e2, 1.0e3, 1.0e4, 1.0e5):
        data.add_data(photon=n, Qi=1.0e5 + n)
    return data


def _axes(plot):
    return plot.plot.fig.axes[0]


def test_axes_are_linear_by_default(plotWidget):
    plotWidget.setData(_decades())
    assert _axes(plotWidget).get_xscale() == 'linear'
    assert _axes(plotWidget).get_yscale() == 'linear'


@pytest.mark.parametrize('x, y', [('log', 'linear'), ('linear', 'log'),
                                  ('log', 'log')])
def test_each_axis_can_be_switched(plotWidget, x, y):
    plotWidget.setData(_decades())
    plotWidget._axisScalesFromToolBar(x, y)
    assert _axes(plotWidget).get_xscale() == x
    assert _axes(plotWidget).get_yscale() == y


def test_switching_back_to_linear(plotWidget):
    plotWidget.setData(_decades())
    plotWidget._axisScalesFromToolBar('log', 'log')
    plotWidget._axisScalesFromToolBar('linear', 'linear')
    assert _axes(plotWidget).get_xscale() == 'linear'
    assert _axes(plotWidget).get_yscale() == 'linear'


def test_the_choice_survives_new_data(plotWidget):
    plotWidget.setData(_decades())
    plotWidget._axisScalesFromToolBar('log', 'linear')
    plotWidget.setData(_decades())
    assert _axes(plotWidget).get_xscale() == 'log'


def test_a_log_axis_is_skipped_where_nothing_is_positive(plotWidget):
    """空のプロットを出すより、線形のままの方がよい。"""
    data = DataDict(x=dict(), y=dict(axes=['x']))
    data.validate()
    data.add_data(x=[-3.0, -2.0, -1.0], y=[1.0, 2.0, 3.0])

    plotWidget.setData(data)
    plotWidget._axisScalesFromToolBar('log', 'log')
    assert _axes(plotWidget).get_xscale() == 'linear'   # 負だけ
    assert _axes(plotWidget).get_yscale() == 'log'      # こちらは正


def test_two_dimensional_plots_are_left_alone(plotWidget):
    """画像の軸は画素の並びなので、対数にしても意味がない。"""
    from plottr.plot.mpl.plotting import PlotType

    data = DataDict(x=dict(), y=dict(), z=dict(axes=['x', 'y']))
    data.validate()
    for x in (1.0, 2.0):
        for y in (1.0, 2.0):
            data.add_data(x=x, y=y, z=x * y)

    plotWidget.setData(data.expand())
    plotWidget._axisScalesFromToolBar('log', 'log')
    # 落ちないこと、そして 2 次元表示では軸をいじらないこと
    axes = plotWidget.plot.fig.axes
    if axes and plotWidget.plotType in (PlotType.image, PlotType.colormesh,
                                        PlotType.scatter2d):
        assert axes[0].get_xscale() == 'linear'


def test_the_toolbar_shows_the_current_scales(plotWidget):
    bar = plotWidget.plotOptionsToolBar
    bar.setAxisScales('log', 'linear')
    assert bar.xScaleBox.currentData() == 'log'
    assert bar.yScaleBox.currentData() == 'linear'

    emitted = []
    bar.axisScaleSelected.connect(lambda x, y: emitted.append((x, y)))
    bar.xScaleBox.setCurrentIndex(bar.xScaleBox.findData('linear'))
    assert emitted and emitted[-1] == ('linear', 'linear')
