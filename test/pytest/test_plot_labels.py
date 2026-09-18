"""Tests for editing what the figure says: title, axis labels, legend.

The figure follows the data until it is told not to.  Every field here is
"empty means automatic", and the tests are mostly about that: typing into one
field replaces that one piece and leaves the rest alone, and clearing it goes
back to what the data said.

The legend is keyed by the label matplotlib ended up with, not by the column
name, because a complex trace drawn as Re and Im is two entries and both have
to be nameable.
"""

import numpy as np
import pytest

from plottr.data.datadict import DataDict
from plottr.plot.base import PlotWidgetContainer
from plottr.plot.mpl.autoplot import (LEGEND_AUTO, LEGEND_NONE, LEGEND_OUTSIDE,
                                      AutoPlot, PlotLabels, renderableText)


def _twoTraces(points=11):
    """Two quantities on one axis -- the case that gets a legend."""
    x = np.linspace(1.0, 100.0, points)
    data = DataDict(
        photon=dict(unit=''),
        Qi=dict(axes=['photon']),
        Qc=dict(axes=['photon']),
    )
    data.validate()
    data.add_data(photon=x, Qi=1.0e5 / x, Qc=2.0e5 / x)
    return data


def _oneTrace(points=11):
    x = np.linspace(1.0, 100.0, points)
    data = DataDict(photon=dict(unit=''), Qi=dict(axes=['photon']))
    data.validate()
    data.add_data(photon=x, Qi=1.0e5 / x)
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


def _legendTexts(plot):
    legend = _axes(plot).get_legend()
    if legend is None:
        return None
    return [t.get_text() for t in legend.get_texts()]


# ---------------------------------------------------------------------------
# タイトル
# ---------------------------------------------------------------------------


def test_the_title_is_the_datasets_until_one_is_typed(plotWidget):
    data = _oneTrace()
    data.add_meta('title', '/data/2026-09-18T120000_ab-CD32_r1/data.ddh5')
    plotWidget.setData(data)
    assert plotWidget.plot.fig._suptitle.get_text().endswith('data.ddh5')

    plotWidget.plotLabels.title = 'CD32, 20 mK'
    plotWidget._plotData()
    assert plotWidget.plot.fig._suptitle.get_text() == 'CD32, 20 mK'

    # 空にすればデータセットのものへ戻る
    plotWidget.plotLabels.title = ''
    plotWidget._plotData()
    assert plotWidget.plot.fig._suptitle.get_text().endswith('data.ddh5')


def test_the_title_can_be_turned_off(plotWidget):
    """自動のタイトルはファイルのフルパスで、発表用の図には長すぎる。"""
    data = _oneTrace()
    data.add_meta('title', '/data/2026-09-18T120000_ab-CD32_r1/data.ddh5')
    plotWidget.setData(data)

    plotWidget.plotLabels.showTitle = False
    plotWidget._plotData()
    assert plotWidget.plot.fig._suptitle.get_text() == ''


# ---------------------------------------------------------------------------
# 軸ラベル
# ---------------------------------------------------------------------------


def test_axis_labels_are_the_columns_until_typed(plotWidget):
    plotWidget.setData(_oneTrace())
    assert _axes(plotWidget).get_xlabel() == 'photon'

    plotWidget.plotLabels.xLabel = 'Photon number'
    plotWidget.plotLabels.yLabel = 'Quality factor'
    plotWidget._plotData()
    assert _axes(plotWidget).get_xlabel() == 'Photon number'
    assert _axes(plotWidget).get_ylabel() == 'Quality factor'

    plotWidget.plotLabels.xLabel = ''
    plotWidget._plotData()
    assert _axes(plotWidget).get_xlabel() == 'photon'


def test_the_automatic_labels_are_offered_as_placeholders(plotWidget):
    """空欄が何の代わりなのかが、打つ前に分かること。"""
    data = _oneTrace()
    data.add_meta('title', 'somewhere/data.ddh5')
    plotWidget.setData(data)

    widget = plotWidget.plotOptionsToolBar.labelsWidget
    assert widget.xLabel.placeholderText() == 'photon'
    assert widget.title.placeholderText() == 'somewhere/data.ddh5'

    # 自分で入れた値が「自動の値」に化けないこと
    plotWidget.plotLabels.xLabel = 'Photon number'
    plotWidget._plotData()
    assert widget.xLabel.placeholderText() == 'photon'


# ---------------------------------------------------------------------------
# 凡例
# ---------------------------------------------------------------------------


def test_automatic_puts_a_legend_only_where_one_is_needed(plotWidget):
    plotWidget.setData(_oneTrace())
    assert plotWidget.plotLabels.legendLocation == LEGEND_AUTO
    assert _legendTexts(plotWidget) is None    # 1 本なら y ラベルで足りる

    plotWidget.setData(_twoTraces())
    assert _legendTexts(plotWidget) == ['Qi', 'Qc']


def test_a_legend_can_be_asked_for_where_it_is_not_needed(plotWidget):
    plotWidget.setData(_oneTrace())
    plotWidget.plotLabels.legendLocation = 'lower left'
    plotWidget._plotData()
    assert _legendTexts(plotWidget) == ['Qi']


def test_the_legend_can_be_hidden(plotWidget):
    plotWidget.setData(_twoTraces())
    plotWidget.plotLabels.legendLocation = LEGEND_NONE
    plotWidget._plotData()
    assert _legendTexts(plotWidget) is None


def test_the_legend_can_go_outside_the_axes(plotWidget):
    """6 本重ねると、何も隠さない角がもう無い。"""
    plotWidget.setData(_twoTraces())
    plotWidget.plotLabels.legendLocation = LEGEND_OUTSIDE
    plotWidget._plotData()

    legend = _axes(plotWidget).get_legend()
    assert legend is not None
    # 軸の右外に置かれていること
    assert legend.get_bbox_to_anchor() is not None


def test_legend_entries_can_be_renamed(plotWidget):
    plotWidget.setData(_twoTraces())
    plotWidget.plotLabels.legendNames = {'Qi': 'internal', 'Qc': 'coupling'}
    plotWidget._plotData()
    assert _legendTexts(plotWidget) == ['internal', 'coupling']

    # 名前を消せば元に戻る
    plotWidget.plotLabels.legendNames = {}
    plotWidget._plotData()
    assert _legendTexts(plotWidget) == ['Qi', 'Qc']


def test_the_toolbar_lists_the_traces_that_are_in_the_plot(plotWidget):
    plotWidget.setData(_twoTraces())
    table = plotWidget.plotOptionsToolBar.labelsWidget.entries
    names = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert names == ['Qi', 'Qc']


def test_renaming_from_the_table_reaches_the_plot(plotWidget):
    from plottr import QtWidgets

    plotWidget.setData(_twoTraces())
    widget = plotWidget.plotOptionsToolBar.labelsWidget
    widget.entries.setItem(0, 1, QtWidgets.QTableWidgetItem('internal'))

    assert plotWidget.plotLabels.legendNames == {'Qi': 'internal'}
    assert _legendTexts(plotWidget) == ['internal', 'Qc']


# ---------------------------------------------------------------------------
# 打ち間違えても図が消えないこと。
#
# ラベルには数式を入れたくなる（`$Q_i$`）。matplotlib は解釈できない `$...$` を
# **描画中に**例外にするので、図が出なくなる（ウィンドウが空になる）。
# ---------------------------------------------------------------------------


def test_broken_maths_becomes_plain_text():
    assert renderableText('$Q_i$') == '$Q_i$'
    assert renderableText('no maths here') == 'no maths here'
    broken = renderableText('$Q_i')
    assert '$' in broken and broken != '$Q_i'


def test_a_broken_label_does_not_take_the_figure_down(plotWidget):
    plotWidget.setData(_oneTrace())
    plotWidget.plotLabels.yLabel = r'$\frac{'      # 閉じていない
    plotWidget._plotData()
    plotWidget.plot.fig.canvas.draw()              # ここで落ちていた
    assert plotWidget.plot.fig.axes


# ---------------------------------------------------------------------------
# 設定そのもの
# ---------------------------------------------------------------------------


def test_defaults_are_all_automatic():
    labels = PlotLabels()
    assert labels.title == '' and labels.showTitle
    assert labels.xLabel == '' and labels.yLabel == ''
    assert labels.legendLocation == LEGEND_AUTO
    assert labels.legendNames == {}
    assert labels.nameFor('Qi') == 'Qi'


def test_hidden_means_no_legend_keywords():
    labels = PlotLabels()
    labels.legendLocation = LEGEND_NONE
    assert labels.legendKeywords() is None

    labels.legendLocation = 'lower left'
    assert labels.legendKeywords()['loc'] == 'lower left'
