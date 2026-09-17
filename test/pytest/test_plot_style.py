"""Tests for how measured traces and their fits are drawn.

A fit curve and the trace it belongs to have to read as one thing: same color,
the curve on top of the markers (a fit hidden under the points cannot be
judged), and no markers on the curve itself -- the points on the plot should be
the ones that were actually measured.

The pairing is by name only: ``<trace>_fit`` is the fit of ``<trace>``.  That
covers what the measurement scripts save (``s11_raw_fit``, ``s11_cor_fit``,
and the older ``s21_fit_normalized``).
"""

import numpy as np
import pytest

from plottr.data.datadict import DataDict
from plottr.plot.base import (PlotWidgetContainer, ComplexRepresentation,
                              fitSourceName, isFitData, sortFitsLast)
from plottr.plot.mpl.autoplot import AutoPlot, FIT_COLOR_AUTO


def _dataset(points=21):
    f = np.linspace(9.99e9, 10.01e9, points)
    z = 1 - 0.8 / (1 + 2j * 5000 * (f - 1e10) / 1e10)
    data = DataDict(
        frequency=dict(unit='Hz'),
        s11_raw=dict(axes=['frequency']),
        s11_raw_fit=dict(axes=['frequency']),
    )
    data.validate()
    data.add_data(frequency=f, s11_raw=z, s11_raw_fit=z)
    return data


def _lines(plot):
    return plot.plot.fig.axes[0].get_lines()


#: Widgets are kept here for the session.  Dropping the container deletes the
#: Qt tree under it -- including the toolbar's actions -- while the plot widget
#: is still being used, so they are not handed to qtbot for teardown.
_WIDGETS = []


@pytest.fixture
def plotWidget(qtbot):
    """An AutoPlot to draw into."""
    container = PlotWidgetContainer()
    plot = AutoPlot(container)
    _WIDGETS.append((container, plot))
    return plot


def test_fit_columns_are_recognised_by_name():
    data = _dataset()
    assert fitSourceName(data, 's11_raw_fit') == 's11_raw'
    assert fitSourceName(data, 's11_raw') is None
    assert isFitData(data, 's11_raw_fit')

    # 名前が合っても、相手がデータセットに無ければフィットとは見なさない。
    assert fitSourceName(data, 'nothing_fit') is None


@pytest.mark.parametrize('name, source', [
    ('s11_raw_fit', 's11_raw'),
    ('s11_cor_fit', 's11_cor'),
    ('s21_fit_normalized', 's21_normalized'),   # 古い cw_punchout の命名
])
def test_the_naming_conventions_we_save(name, source):
    data = DataDict(x=dict(), **{source: dict(axes=['x']), name: dict(axes=['x'])})
    data.validate()
    data.add_data(x=[0.0], **{source: [1.0], name: [1.0]})
    assert fitSourceName(data, name) == source


def test_fits_are_drawn_after_their_data():
    data = _dataset()
    assert sortFitsLast(data, ['s11_raw_fit', 's11_raw']) == ['s11_raw', 's11_raw_fit']


def test_fit_sits_on_top_and_takes_the_color_of_its_data(plotWidget):
    plot = plotWidget
    plot.setData(_dataset())

    lines = {str(line.get_label()): line for line in _lines(plot)}
    for part in ('Real', 'Imag'):
        measured = lines[f's11_raw ({part})']
        fit = lines[f's11_raw_fit ({part})']
        assert fit.get_color() == measured.get_color()
        assert fit.get_zorder() > measured.get_zorder()
        assert fit.get_marker() in ('', 'None', None)
        assert measured.get_marker() not in ('', 'None', None)


def test_point_size_and_line_widths_are_settable(plotWidget):
    plot = plotWidget
    plot.setData(_dataset())

    plot.plotStyle.markerSize = 9.0
    plot.plotStyle.lineWidth = 0.5
    plot.plotStyle.fitLineWidth = 3.0
    plot._plotStyleFromToolBar()

    lines = {str(line.get_label()): line for line in _lines(plot)}
    measured = lines['s11_raw (Real)']
    fit = lines['s11_raw_fit (Real)']
    assert measured.get_markersize() == 9.0
    assert measured.get_linewidth() == 0.5
    assert fit.get_linewidth() == 3.0


def test_point_size_zero_hides_the_markers(plotWidget):
    plot = plotWidget
    plot.setData(_dataset())
    plot.plotStyle.markerSize = 0.0
    plot._plotStyleFromToolBar()

    measured = {str(l.get_label()): l for l in _lines(plot)}['s11_raw (Real)']
    assert measured.get_marker() in ('', 'None', None)


def test_an_explicit_fit_color_is_used_for_every_fit(plotWidget):
    plot = plotWidget
    plot.setData(_dataset())
    plot.plotStyle.fitColor = '#d62728'
    plot._plotStyleFromToolBar()

    lines = {str(line.get_label()): line for line in _lines(plot)}
    assert lines['s11_raw_fit (Real)'].get_color() == '#d62728'
    assert lines['s11_raw_fit (Imag)'].get_color() == '#d62728'
    # データの色は変わらない
    assert lines['s11_raw (Real)'].get_color() != '#d62728'

    # 既定に戻せば、また相手の色になる。
    plot.plotStyle.fitColor = FIT_COLOR_AUTO
    plot._plotStyleFromToolBar()
    lines = {str(line.get_label()): line for line in _lines(plot)}
    assert lines['s11_raw_fit (Real)'].get_color() == lines['s11_raw (Real)'].get_color()


def test_real_data_still_plots(plotWidget):
    """フィット列が無い、実数だけのデータでも従来どおり描ける。"""
    data = DataDict(x=dict(unit='s'), y=dict(axes=['x']))
    data.validate()
    data.add_data(x=np.arange(5.0), y=np.arange(5.0))

    plotWidget.setData(data)
    assert len(_lines(plotWidget)) == 1


# ---------------------------------------------------------------------------
# ツールバーの Style パネル。
# ---------------------------------------------------------------------------


def test_style_widget_edits_the_style_in_place(qtbot):
    from plottr.plot.mpl.autoplot import PlotStyle, PlotStyleWidget

    style = PlotStyle()
    widget = PlotStyleWidget()
    qtbot.addWidget(widget)
    widget.setStyle(style)

    changes = []
    widget.changed.connect(lambda: changes.append(True))

    widget.pointSize.setValue(7.5)
    assert style.markerSize == 7.5
    widget.lineWidth.setValue(0.0)
    assert style.lineWidth == 0.0
    widget.fitLineWidth.setValue(2.5)
    assert style.fitLineWidth == 2.5
    assert changes

    index = widget.fitColor.findData('#d62728')
    widget.fitColor.setCurrentIndex(index)
    assert style.fitColor == '#d62728'

    # 既定に戻せる
    widget.fitColor.setCurrentIndex(widget.fitColor.findData(FIT_COLOR_AUTO))
    assert style.fitColor == FIT_COLOR_AUTO


def test_style_widget_shows_the_values_it_is_given(qtbot):
    from plottr.plot.mpl.autoplot import PlotStyle, PlotStyleWidget

    style = PlotStyle()
    style.markerSize, style.fitLineWidth, style.fitColor = 4.0, 3.0, '#123456'
    widget = PlotStyleWidget()
    qtbot.addWidget(widget)
    widget.setStyle(style)

    assert widget.pointSize.value() == 4.0
    assert widget.fitLineWidth.value() == 3.0
    assert widget.fitColor.currentData() == '#123456'
    # フォームを埋めただけで値が書き換わっていないこと
    assert style.markerSize == 4.0 and style.fitColor == '#123456'


# ---------------------------------------------------------------------------
# 図の中の文字は、キャンバスの大きさに合わせる。
# ---------------------------------------------------------------------------


def test_figure_font_scale_follows_the_canvas():
    from plottr.plot.mpl.widgets import (FIGURE_FONT_RANGE,
                                         REFERENCE_FIGURE_INCHES,
                                         figureFontScale)

    assert figureFontScale(*REFERENCE_FIGURE_INCHES) == pytest.approx(1.0)
    assert figureFontScale(9.0, 6.0) > 1.0
    assert figureFontScale(2.0, 1.5) < 1.0
    # 横に広いだけの図では大きくしない
    assert figureFontScale(20.0, 3.0) == figureFontScale(4.5, 3.0)
    # 上下限
    low, high = FIGURE_FONT_RANGE
    assert figureFontScale(0.1, 0.1) >= low
    assert figureFontScale(100.0, 100.0) <= high
    assert figureFontScale(0.0, 0.0) == 1.0


def test_plot_text_is_resized_with_the_canvas(plotWidget):
    """再描画をまたいでも、今のキャンバスの大きさに合った文字になる。"""
    data = DataDict(x=dict(unit='s'), y=dict(axes=['x']))
    data.validate()
    data.add_data(x=np.arange(10.0), y=np.arange(10.0) ** 2)
    plotWidget.setData(data)

    plot = plotWidget.plot
    plot.fig.set_size_inches(4.5, 3.0)
    plot.applyFontSize()
    small = plot.fig.axes[0].xaxis.label.get_fontsize()

    plot.fig.set_size_inches(12.0, 7.0)
    plot.applyFontSize()
    large = plot.fig.axes[0].xaxis.label.get_fontsize()
    assert large > small

    # 描き直しても大きいまま（rcParams が先に効く）
    plotWidget.setData(data)
    assert plot.fig.axes[0].xaxis.label.get_fontsize() == pytest.approx(large, rel=0.05)
