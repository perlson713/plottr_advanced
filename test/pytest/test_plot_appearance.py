"""Tests for the rest of what the toolbar can do to a figure.

Axis ranges, the look of one trace at a time, the color scale of a 2-D plot,
writing the figure out at a chosen size, and keeping the whole lot in a file.

The thread through all of it is the same as for the labels: **empty means
automatic**.  A blank field, an empty string, a ``None`` limit all mean
"whatever the data says", and the tests are mostly about a field going back to
that when it is cleared -- a figure that cannot be made automatic again is a
figure that has to be closed and reopened.
"""

import json

import numpy as np
import pytest

from plottr.data.datadict import DataDict, MeshgridDataDict
from plottr.plot.base import PlotWidgetContainer
from plottr.plot.mpl.autoplot import AutoPlot, exportFigure
from plottr.plot.mpl.appearance import (APPEARANCE_VERSION, FigureExport,
                                        PlotAxes, PlotColors, PlotLabels,
                                        PlotStyle, TraceStyle,
                                        appearanceToDict, applyAppearance,
                                        parseNumber)
from plottr.plot.mpl.plotting import PlotType


def _twoTraces(points=11):
    x = np.linspace(1.0, 100.0, points)
    data = DataDict(photon=dict(unit=''), Qi=dict(axes=['photon']),
                    Qc=dict(axes=['photon']))
    data.validate()
    data.add_data(photon=x, Qi=1.0e5 / x, Qc=2.0e5 / x)
    return data


def _grid(nx=7, ny=5):
    x = np.linspace(9.99e9, 1.001e10, nx)
    y = np.linspace(-30.0, 10.0, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    data = MeshgridDataDict(
        frequency=dict(values=xx, unit='Hz'),
        power=dict(values=yy, unit='dBm'),
        mag=dict(values=np.abs(np.sin(xx / 1e9) + yy / 100.0),
                 axes=['frequency', 'power']),
    )
    data.validate()
    return data


#: See test_plot_style.py -- the widgets outlive the test on purpose.
_WIDGETS = []


@pytest.fixture
def plotWidget(qtbot):
    container = PlotWidgetContainer()
    plot = AutoPlot(container)
    _WIDGETS.append((container, plot))
    return plot


def _axes(plot):
    return plot.plot.fig.axes[0]


# ---------------------------------------------------------------------------
# 1. 軸の範囲
# ---------------------------------------------------------------------------


def test_axis_limits_are_the_datas_until_typed(plotWidget):
    plotWidget.setData(_twoTraces())
    automatic = _axes(plotWidget).get_ylim()

    plotWidget.plotAxes.yMin, plotWidget.plotAxes.yMax = 0.0, 5.0e5
    plotWidget._plotData()
    assert _axes(plotWidget).get_ylim() == (0.0, 5.0e5)

    plotWidget.plotAxes.yMin = plotWidget.plotAxes.yMax = None
    plotWidget._plotData()
    assert _axes(plotWidget).get_ylim() == pytest.approx(automatic)


def test_one_end_of_a_range_can_be_left_automatic(plotWidget):
    plotWidget.setData(_twoTraces())
    plotWidget.plotAxes.yMin = 0.0
    plotWidget._plotData()

    bottom, top = _axes(plotWidget).get_ylim()
    assert bottom == 0.0
    assert top > 1.0e5           # データが決めたまま


def test_the_range_can_be_taken_from_the_view(plotWidget):
    """ズームで目で合わせてから、それを数字として残す。"""
    plotWidget.setData(_twoTraces())
    _axes(plotWidget).set_xlim(2.0, 20.0)
    plotWidget._limitsFromView()

    assert plotWidget.plotAxes.xMin == pytest.approx(2.0)
    assert plotWidget.plotAxes.xMax == pytest.approx(20.0)
    assert _axes(plotWidget).get_xlim() == pytest.approx((2.0, 20.0))
    # フォームにも出ていること（打ち直せる形で）
    assert plotWidget.plotOptionsToolBar.axesWidget.limits['xMin'].text()


def test_the_form_clears_the_range(plotWidget):
    plotWidget.setData(_twoTraces())
    plotWidget.plotAxes.xMin, plotWidget.plotAxes.xMax = 2.0, 20.0
    widget = plotWidget.plotOptionsToolBar.axesWidget
    widget.setAxes(plotWidget.plotAxes)
    widget._clearLimits()

    assert plotWidget.plotAxes.isEmpty()


def test_a_half_typed_number_reads_as_automatic():
    """打っている途中で図が変わらないこと。"""
    assert parseNumber('') is None
    assert parseNumber('-') is None
    assert parseNumber('1e') is None
    assert parseNumber(' 2.5 ') == 2.5


# ---------------------------------------------------------------------------
# 2. 曲線ごとの色・点・線
# ---------------------------------------------------------------------------


def test_a_trace_can_be_given_its_own_colour_and_marker(plotWidget):
    plotWidget.setData(_twoTraces())
    plotWidget.plotStyle.traceStyle('Qi').color = '#d62728'
    plotWidget.plotStyle.traceStyle('Qi').marker = 's'
    plotWidget.plotStyle.traceStyle('Qc').lineStyle = '--'
    plotWidget._plotData()

    lines = {line.get_label(): line for line in _axes(plotWidget).get_lines()}
    assert lines['Qi'].get_color() == '#d62728'
    assert lines['Qi'].get_marker() == 's'
    assert lines['Qc'].get_linestyle() == '--'
    # 指定していない方は色の順番のまま
    assert lines['Qc'].get_color() != '#d62728'


def test_a_trace_can_be_left_with_no_markers_or_no_line(plotWidget):
    plotWidget.setData(_twoTraces())
    plotWidget.plotStyle.traceStyle('Qi').marker = 'none'
    plotWidget.plotStyle.traceStyle('Qc').lineStyle = 'none'
    plotWidget._plotData()

    lines = {line.get_label(): line for line in _axes(plotWidget).get_lines()}
    assert lines['Qi'].get_marker() in ('', 'None')
    assert lines['Qc'].get_linestyle() in ('none', 'None')


def test_an_empty_trace_style_changes_nothing():
    style = PlotStyle()
    assert TraceStyle().options() == {}
    assert TraceStyle().isEmpty()
    # 何も指定していない曲線は、全体の設定そのまま
    assert style.dataOptions('Qi') == style.dataOptions()


def test_the_fit_follows_the_colour_its_trace_was_given(plotWidget):
    """フィットは自分が属する曲線と同じ色 — 手で選んだ色でも同じこと。"""
    data = DataDict(x=dict(), s11=dict(axes=['x']), s11_fit=dict(axes=['x']))
    data.validate()
    values = np.linspace(0.0, 1.0, 9)
    data.add_data(x=values, s11=values, s11_fit=values)

    plotWidget.setData(data)
    label = _axes(plotWidget).get_lines()[0].get_label()
    plotWidget.plotStyle.traceStyle(label).color = '#2ca02c'
    plotWidget._plotData()

    colors = {line.get_color() for line in _axes(plotWidget).get_lines()}
    assert colors == {'#2ca02c'}


# ---------------------------------------------------------------------------
# 3. 書き出し
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('fmt', ['pdf', 'png', 'svg'])
def test_the_figure_is_written_at_the_size_asked_for(plotWidget, tmp_path, fmt):
    plotWidget.setData(_twoTraces())
    settings = FigureExport()
    settings.format = fmt
    settings.width, settings.height = 86.0, 60.0

    before = tuple(plotWidget.plot.fig.get_size_inches())
    path = tmp_path / f'figure.{fmt}'
    exportFigure(plotWidget.plot.fig, str(path), settings,
                 plotWidget.plot.applyFontSize)

    assert path.stat().st_size > 0
    # 図は元の大きさに戻っていること（窓が跳ねない）
    assert tuple(plotWidget.plot.fig.get_size_inches()) == pytest.approx(before)


def test_the_export_dialog_shows_and_reads_back_the_settings(plotWidget):
    """ダイアログが作れて、値が往復すること。

    最初の版は欄を `self.width` / `self.height` に持っていた — QWidget 自身の
    メソッド名で、Qt がそれを呼ぶので、開いた瞬間に
    `'QDoubleSpinBox' object is not callable` で落ちていた。
    """
    from plottr.plot.mpl.autoplot import ExportDialog

    settings = FigureExport()
    dialog = ExportDialog(settings, plotWidget)
    dialog.show()                       # レイアウトはここで走る
    assert dialog.widthBox.value() == settings.width

    dialog.widthBox.setValue(86.0)
    dialog.formatBox.setCurrentIndex(dialog.formatBox.findData('svg'))
    dialog.transparent.setChecked(True)
    dialog.apply()
    dialog.close()

    assert settings.width == 86.0
    assert settings.format == 'svg'
    assert settings.transparent


def test_an_export_preset_fills_in_the_size(plotWidget):
    from plottr.plot.mpl.autoplot import ExportDialog

    dialog = ExportDialog(FigureExport(), plotWidget)
    name, width, height = FigureExport.PRESETS[0]
    dialog.preset.setCurrentIndex(dialog.preset.findText(name))
    assert dialog.widthBox.value() == width
    assert dialog.heightBox.value() == height
    dialog.close()


def test_millimetres_become_inches():
    settings = FigureExport()
    settings.width, settings.height = 254.0, 25.4
    assert settings.inches() == pytest.approx((10.0, 1.0))


def test_a_transparent_export_does_not_paint_the_background():
    settings = FigureExport()
    assert settings.saveKeywords()['facecolor'] == 'white'
    settings.transparent = True
    assert 'facecolor' not in settings.saveKeywords()


# ---------------------------------------------------------------------------
# 4. 2 次元表示の色
# ---------------------------------------------------------------------------


def test_the_colormap_and_range_reach_the_image(plotWidget):
    plotWidget.setData(_grid())
    plotWidget._plotTypeFromToolBar(PlotType.image)

    plotWidget.plotColors.colormap = 'inferno'
    plotWidget.plotColors.vMin, plotWidget.plotColors.vMax = 0.2, 0.8
    plotWidget._plotData()

    images = _axes(plotWidget).get_images()
    assert images, 'no image was drawn'
    assert images[0].cmap.name == 'inferno'
    assert images[0].get_clim() == (0.2, 0.8)


def test_the_colorbar_can_be_relabelled(plotWidget):
    plotWidget.setData(_grid())
    plotWidget._plotTypeFromToolBar(PlotType.image)
    plotWidget.plotLabels.colorbarLabel = 'reflection'
    plotWidget._plotData()

    # カラーバーは最後に足された軸
    assert plotWidget.plot.fig.axes[-1].get_ylabel() == 'reflection'


def test_an_empty_colour_setting_changes_nothing():
    assert PlotColors().keywords() == {}


def test_the_colour_button_is_only_there_for_two_dimensional_plots(plotWidget):
    plotWidget.setData(_twoTraces())
    assert not plotWidget.plotOptionsToolBar.colorsButtonAction.isVisible()

    plotWidget.setData(_grid())
    plotWidget._plotTypeFromToolBar(PlotType.image)
    assert plotWidget.plotOptionsToolBar.colorsButtonAction.isVisible()


# ---------------------------------------------------------------------------
# 5. 体裁を保存して、次のデータセットに掛ける
#
# これが無いと上の全部が使い捨てになる: 10 個の欄を埋めて作った図を、
# 次のファイルでもう一度打ち直すことになる。
# ---------------------------------------------------------------------------


def _filled():
    style, labels = PlotStyle(), PlotLabels()
    axes, colors, export = PlotAxes(), PlotColors(), FigureExport()

    style.markerSize = 7.0
    style.fitColor = '#000000'
    style.traceStyle('Qi').color = '#d62728'
    labels.title = 'CD32, 20 mK'
    labels.xLabel = 'photon number'
    labels.legendLocation = 'outside'
    labels.legendNames = {'Qi': 'internal'}
    axes.yMin, axes.yMax = 1.0e4, 1.0e6
    colors.colormap = 'inferno'
    colors.vMax = 0.8
    export.width, export.format = 86.0, 'svg'
    return style, labels, axes, colors, export


def test_an_appearance_survives_a_round_trip_through_a_file(tmp_path):
    saved = appearanceToDict(*_filled(), xScale='log', yScale='linear')
    path = tmp_path / 'appearance.json'
    path.write_text(json.dumps(saved), encoding='utf-8')

    style, labels = PlotStyle(), PlotLabels()
    axes, colors, export = PlotAxes(), PlotColors(), FigureExport()
    xScale, yScale = applyAppearance(
        json.loads(path.read_text(encoding='utf-8')),
        style, labels, axes, colors, export)

    assert (xScale, yScale) == ('log', 'linear')
    assert style.markerSize == 7.0
    assert style.traces['Qi'].color == '#d62728'
    assert labels.title == 'CD32, 20 mK'
    assert labels.legendNames == {'Qi': 'internal'}
    assert axes.yMin == 1.0e4 and axes.yMax == 1.0e6
    assert colors.colormap == 'inferno' and colors.vMax == 0.8
    assert export.width == 86.0 and export.format == 'svg'


def test_a_file_from_a_newer_plottr_is_refused_not_half_applied():
    values = appearanceToDict(*_filled())
    values['version'] = APPEARANCE_VERSION + 1

    labels = PlotLabels()
    with pytest.raises(ValueError, match='newer'):
        applyAppearance(values, PlotStyle(), labels, PlotAxes(), PlotColors(),
                        FigureExport())
    assert labels.title == ''    # 途中まで当たっていない


def test_a_file_missing_a_section_leaves_that_part_alone():
    labels = PlotLabels()
    labels.title = 'kept'
    axes = PlotAxes()

    applyAppearance({'axes': {'yMin': 3.0}}, PlotStyle(), labels, axes,
                    PlotColors(), FigureExport())
    assert labels.title == 'kept'
    assert axes.yMin == 3.0


def test_loading_an_appearance_reaches_the_plot_and_the_forms(plotWidget):
    plotWidget.setData(_twoTraces())
    style, labels, axes, colors, export = _filled()
    values = appearanceToDict(style, labels, axes, colors, export,
                              xScale='log', yScale='linear')

    plotWidget.applyAppearanceValues(values)

    assert plotWidget.plot.fig._suptitle.get_text() == 'CD32, 20 mK'
    assert _axes(plotWidget).get_xlabel() == 'photon number'
    assert _axes(plotWidget).get_ylim() == (1.0e4, 1.0e6)
    assert _axes(plotWidget).get_xscale() == 'log'
    # フォームにも出ていること
    bar = plotWidget.plotOptionsToolBar
    assert bar.labelsWidget.title.text() == 'CD32, 20 mK'
    assert bar.axesWidget.limits['yMin'].text()
    assert bar.xScaleBox.currentData() == 'log'


def test_everything_can_be_put_back_to_automatic(plotWidget):
    plotWidget.setData(_twoTraces())
    style, labels, axes, colors, export = _filled()
    plotWidget.applyAppearanceValues(
        appearanceToDict(style, labels, axes, colors, export, 'log', 'log'))

    plotWidget._resetAppearance()

    assert plotWidget.plotLabels.title == ''
    assert plotWidget.plotAxes.isEmpty()
    assert plotWidget.plotStyle.traces == {}
    assert plotWidget.plotColors.colormap == ''
    assert (plotWidget.xScale, plotWidget.yScale) == ('linear', 'linear')
    assert _axes(plotWidget).get_xscale() == 'linear'
    assert plotWidget.plotOptionsToolBar.labelsWidget.title.text() == ''
