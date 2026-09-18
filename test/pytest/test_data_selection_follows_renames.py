"""Tests for keeping the data selection across a change of dataset shape.

The overlay of `Qi` from several datasets did not appear.  `Compare datasets`
renames `Qi` to `Qi [CD32_r1]` and `Qi [CD32_r2]`, the selection tree rebuilt
itself and lost the selection, and with nothing selected the node produced no
output at all -- so the window kept showing the *previous* figure.  Adding a
dataset looked like it did nothing.

Two things had to hold, and both are checked here: the selection follows the
rename, and nothing selected means an empty plot rather than a stale one.
"""

import numpy as np
import pytest

from plottr.data.datadict import DataDict
from plottr.gui.data_display import matchSelection, unlabelledName
from plottr.node.data_selector import DataSelector
from plottr.node.tools import linearFlowchart


def test_a_label_is_taken_back_out_of_a_name():
    assert unlabelledName('Qi [CD32_r1]') == 'Qi'
    # ラベルは `_err` の前に入るので、戻すときも付け直す
    assert unlabelledName('Qi [CD32_r1]_err') == 'Qi_err'
    assert unlabelledName('Qi') == 'Qi'
    assert unlabelledName('weird [name') == 'weird [name'


def test_selection_follows_a_rename():
    available = ['Qi [A]', 'Qi [A]_err', 'Qi [B]', 'Qi [B]_err']
    # Qi を見ていた -> 両方の Qi を見たい。エラーバーの列は曲線として選ばない
    assert matchSelection(['Qi'], available) == ['Qi [A]', 'Qi [B]']


def test_selection_comes_back_when_the_labels_go_away():
    assert matchSelection(['Qi [A]', 'Qi [B]'], ['Qi', 'Qi_err', 'Ql']) == ['Qi']


def test_selection_is_kept_when_nothing_was_renamed():
    available = ['Qi', 'Ql', 'Qc']
    assert matchSelection(['Qi', 'Ql'], available) == ['Qi', 'Ql']
    assert matchSelection([], available) == []
    # 消えた列は諦める（作り出さない）
    assert matchSelection(['nothing'], available) == []


def _flowchart():
    DataSelector.useUi = False
    fc = linearFlowchart(('select', DataSelector))
    return fc, fc.nodes()['select']


def _data(names):
    fields = {'power': dict(unit='dBm')}
    for name in names:
        fields[name] = dict(axes=['power'])
    data = DataDict(**fields)
    data.validate()
    data.add_data(power=0.0, **{name: 1.0 for name in names})
    return data


def test_nothing_selected_means_an_empty_plot_not_the_previous_one(qtbot):
    """出力を止めると、下流は前のデータを描いたままになる（それが今回の症状）。"""
    fc, node = _flowchart()
    fc.setInput(dataIn=_data(['Qi', 'Ql']))
    node.selectedData = ['Qi']
    assert fc.output()['dataOut'] is not None

    node.selectedData = []
    assert fc.output()['dataOut'] is None


def test_the_widget_keeps_the_selection_across_a_rebuild(qtbot):
    """木を作り直すとき、選択を落とさない（これが落ちていた）。"""
    from plottr.gui.data_display import DataSelectionWidget

    widget = DataSelectionWidget()
    qtbot.addWidget(widget)
    before = _data(['Qi', 'Ql'])
    widget.setData(before, {name: (1,) for name in before.dependents()})
    widget.setSelectedData(['Qi'])
    assert widget.getSelectedData() == ['Qi']

    emitted = []
    widget.dataSelectionMade.connect(emitted.append)

    after = _data(['Qi [A]', 'Qi [A]_err', 'Qi [B]', 'Qi [B]_err'])
    widget.setData(after, {name: (1,) for name in after.dependents()})

    assert widget.getSelectedData() == ['Qi [A]', 'Qi [B]']
    # 作り直しの途中の「空」ではなく、最終的な選択を 1 回だけ知らせる
    assert emitted and emitted[-1] == ['Qi [A]', 'Qi [B]']
    assert [] not in emitted


def test_a_third_and_fourth_dataset_join_the_selection():
    """選択は「量」に付く。名前を厳密に照合すると、3 つ目以降が無視され、
    重ねられる数に上限があるように見える（実際にそう報告された）。"""
    two = ['Qi [A]', 'Qi [A]_err', 'Qi [B]', 'Qi [B]_err']
    four = two + ['Qi [C]', 'Qi [C]_err', 'Qi [D]', 'Qi [D]_err']

    selected = matchSelection(['Qi'], two)
    assert selected == ['Qi [A]', 'Qi [B]']

    # ここで止まっていた: 既にある名前が見つかるので、増えた分を拾わなかった
    assert matchSelection(selected, four) == ['Qi [A]', 'Qi [B]', 'Qi [C]', 'Qi [D]']


def test_other_quantities_are_not_dragged_in():
    available = ['Qi [A]', 'Qi [B]', 'Ql [A]', 'Ql [B]', 'Qi [A]_err']
    assert matchSelection(['Qi'], available) == ['Qi [A]', 'Qi [B]']
    assert matchSelection(['Qi', 'Ql'], available) == [
        'Qi [A]', 'Qi [B]', 'Ql [A]', 'Ql [B]']
