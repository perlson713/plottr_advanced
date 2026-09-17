"""Tests for drawing several saved datasets in one plot.

The case this exists for: `Qi` against the photon number for two resonators, or
for the same resonator before and after a change.  plottr shows one dataset at
a time, so the node loads the others and merges them.

What must hold: every dataset keeps its own points (nothing is interpolated
onto a common grid), the curves land on one pair of axes, and the legend tells
them apart.  A file that cannot be put in the shape of the current plot is
reported, not guessed at.
"""

import numpy as np
import pytest

from plottr.data.datadict import DataDict
from plottr.data.datadict_storage import datadict_to_hdf5
from plottr.node.tools import linearFlowchart
from plottr.node.dataset_join import (JoinDatasets, datasetLabel, joinDatasets,
                                      reshapeLike, seriesName)

POWERS_A = [10.0, 0.0, -10.0]
POWERS_B = [5.0, -5.0]


def _sweep(powers, qi0=1.0e5, photons=True):
    """A dataset shaped like one of our power sweeps."""
    data = DataDict(
        power=dict(unit='dBm'),
        Qi=dict(axes=['power']),
        Qi_err=dict(axes=['power']),
        photon=dict(axes=['power']),
    )
    data.validate()
    for index, power in enumerate(powers):
        data.add_data(power=power, Qi=qi0 * (index + 1),
                      Qi_err=qi0 * 0.01,
                      photon=10.0 ** (index + 1) if photons else float('nan'))
    return data


def _write(tmp_path, name, data):
    folder = tmp_path / f'2026-09-17T120000_abcd-{name}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'data.ddh5'
    datadict_to_hdf5(data, str(path), groupname='data')
    return str(path)


def _flowchart():
    JoinDatasets.useUi = False
    fc = linearFlowchart(('join', JoinDatasets))
    return fc, fc.nodes()['join']


def test_label_comes_from_the_measurement_name(tmp_path):
    path = _write(tmp_path, 'CD32_r1', _sweep(POWERS_A))
    assert datasetLabel(path) == 'CD32_r1'


def test_error_bars_stay_attached_to_their_partner():
    """`<dep>_err` resolves as the error of `<dep>`, so the label goes first."""
    assert seriesName('Qi', 'A') == 'Qi [A]'
    assert seriesName('Qi_err', 'A') == 'Qi [A]_err'
    assert seriesName('Qi', '') == 'Qi'


def test_join_keeps_every_point_and_fills_the_rest_with_nan():
    a, b = _sweep(POWERS_A), _sweep(POWERS_B, qi0=2.0e5)
    joined, summary = joinDatasets([('A', a), ('B', b)])

    assert joined is not None
    assert joined.axes() == ['power']
    assert set(joined.dependents()) == {
        'Qi [A]', 'Qi [A]_err', 'photon [A]',
        'Qi [B]', 'Qi [B]_err', 'photon [B]'}

    power = np.asarray(joined.data_vals('power'), dtype=float)
    assert list(power) == POWERS_A + POWERS_B

    qi_a = np.asarray(joined.data_vals('Qi [A]'), dtype=float)
    qi_b = np.asarray(joined.data_vals('Qi [B]'), dtype=float)
    assert np.all(np.isfinite(qi_a[:3])) and np.all(np.isnan(qi_a[3:]))
    assert np.all(np.isnan(qi_b[:3])) and np.all(np.isfinite(qi_b[3:]))
    # 測った値はそのまま（内挿も再標本化もしない）
    assert list(qi_a[:3]) == [1.0e5, 2.0e5, 3.0e5]
    assert '2 datasets joined' in summary


def test_error_bars_survive_the_join():
    from plottr.plot.base import errorBarDataName

    joined, _ = joinDatasets([('A', _sweep(POWERS_A)), ('B', _sweep(POWERS_B))])
    assert errorBarDataName(joined, 'Qi [A]') == 'Qi [A]_err'


def test_a_photon_axis_is_rebuilt_for_the_added_file():
    """プロットの横軸が log10_photon なら、足すデータも同じ形にする。"""
    extra = _sweep(POWERS_B)
    shaped, why = reshapeLike(extra, ['log10_photon'])
    assert shaped is not None, why
    assert shaped.axes() == ['log10_photon']
    assert np.allclose(np.asarray(shaped.data_vals('log10_photon'), dtype=float),
                       [1.0, 2.0])


def test_a_file_that_does_not_fit_is_reported_not_guessed():
    without = _sweep(POWERS_B, photons=False)
    del without['photon']
    without.validate()
    shaped, why = reshapeLike(without, ['log10_photon'])
    assert shaped is None
    assert 'photon' in why


def test_node_passes_data_through_until_a_file_is_added(qtbot, tmp_path):
    fc, node = _flowchart()
    fc.setInput(dataIn=_sweep(POWERS_A))
    out = fc.output()['dataOut']
    assert set(out.dependents()) == {'Qi', 'Qi_err', 'photon'}


def test_node_joins_a_file_from_disk(qtbot, tmp_path):
    messages = []
    fc, node = _flowchart()
    node.statusChanged.connect(messages.append)

    path = _write(tmp_path, 'CD32_r2', _sweep(POWERS_B, qi0=2.0e5))
    fc.setInput(dataIn=_sweep(POWERS_A))
    node.files = [path]
    out = fc.output()['dataOut']

    assert 'Qi [CD32_r2]' in out.dependents()
    assert any('2 datasets joined' in m for m in messages)
    power = np.asarray(out.data_vals('power'), dtype=float)
    assert power.size == len(POWERS_A) + len(POWERS_B)


def test_a_broken_file_says_so_and_keeps_the_plot(qtbot, tmp_path):
    messages = []
    fc, node = _flowchart()
    node.statusChanged.connect(messages.append)

    missing = str(tmp_path / 'nope' / 'data.ddh5')
    fc.setInput(dataIn=_sweep(POWERS_A))
    node.files = [missing]
    out = fc.output()['dataOut']

    assert set(out.dependents()) == {'Qi', 'Qi_err', 'photon'}   # 元のまま
    assert messages and messages[-1] != ''


# ---------------------------------------------------------------------------
# 比べる列を絞る。
#
# パワー掃引は 30 列ほど保存する。3 つ重ねると `Data selection` が 100 行になり、
# 見たい 1 本を探すのが仕事になってしまう。
# ---------------------------------------------------------------------------


def _sweepWithExtras(powers):
    """トレースやフィットパラメータまで入った、実際の形に近いデータセット。"""
    freq = np.linspace(9.99e9, 1.001e10, 4)
    data = DataDict(
        frequency=dict(unit='Hz'),
        power=dict(unit='dBm'),
        s11_raw=dict(axes=['frequency', 'power']),
        Qi=dict(axes=['power']),
        Qi_err=dict(axes=['power']),
        Ql=dict(axes=['power']),
        Qc=dict(axes=['power']),
        fr=dict(axes=['power'], unit='Hz'),
        snr=dict(axes=['power']),
        photon=dict(axes=['power']),
    )
    data.validate()
    for index, power in enumerate(powers):
        data.add_data(frequency=freq, power=power,
                      s11_raw=np.ones(freq.size, dtype=complex),
                      Qi=1.0e5 * (index + 1), Qi_err=1.0e3, Ql=5.0e4,
                      Qc=8.0e4, fr=1.0e10, snr=30.0,
                      photon=10.0 ** (index + 1))
    return data


def test_only_the_compared_column_is_brought_over(qtbot, tmp_path):
    from plottr.node.dataset_join import DEFAULT_COLUMNS

    fc, node = _flowchart()
    path = _write(tmp_path, 'CD32_r2', _sweep(POWERS_B, qi0=2.0e5))
    fc.setInput(dataIn=_sweep(POWERS_A))
    node.files = [path]
    out = fc.output()['dataOut']

    assert node.columns == DEFAULT_COLUMNS == 'Qi'
    names = set(out.dependents())
    # 比べる列と、そのエラーバーだけ
    assert names == {'Qi [A]'.replace('A', 'this one'), 'Qi [this one]_err',
                     'Qi [CD32_r2]', 'Qi [CD32_r2]_err'}
    assert not any('photon [' in name for name in names)


def test_the_flood_of_columns_is_what_this_prevents(qtbot, tmp_path):
    fc, node = _flowchart()
    path = _write(tmp_path, 'CD32_r2', _sweepWithExtras(POWERS_B))
    fc.setInput(dataIn=_sweepWithExtras(POWERS_A))

    node.columns = ''            # 全部持ってくる（従来の動き）
    node.files = [path]
    everything = set(fc.output()['dataOut'].dependents())

    node.columns = 'Qi'
    few = set(fc.output()['dataOut'].dependents())

    assert len(few) < len(everything)
    assert len(few) == 4         # Qi と Qi_err が 2 データセットぶん
    assert not any('s11_raw' in name for name in few)


def test_several_columns_can_be_compared(qtbot, tmp_path):
    fc, node = _flowchart()
    path = _write(tmp_path, 'CD32_r2', _sweepWithExtras(POWERS_B))
    fc.setInput(dataIn=_sweepWithExtras(POWERS_A))
    node.columns = 'Qi, Ql'
    node.files = [path]

    names = set(fc.output()['dataOut'].dependents())
    assert any(name.startswith('Ql [') for name in names)
    assert any(name.startswith('Qi [') for name in names)
    assert not any(name.startswith('Qc [') for name in names)


def test_a_column_that_is_not_there_says_so(qtbot, tmp_path):
    messages = []
    fc, node = _flowchart()
    node.statusChanged.connect(messages.append)

    path = _write(tmp_path, 'CD32_r2', _sweep(POWERS_B))
    fc.setInput(dataIn=_sweep(POWERS_A))
    node.columns = 'nothing_like_this'
    node.files = [path]
    out = fc.output()['dataOut']

    assert set(out.dependents()) == {'Qi', 'Qi_err', 'photon'}   # 元のまま
    assert any('nothing_like_this' in m for m in messages)


def test_error_bars_come_along_without_being_asked(qtbot, tmp_path):
    from plottr.node.dataset_join import wantedColumns

    data = _sweepWithExtras(POWERS_A)
    assert wantedColumns(data, ['Qi']) == ['Qi', 'Qi_err']
    assert wantedColumns(data, ['Ql']) == ['Ql']       # 誤差の列が無ければそれだけ
    assert wantedColumns(data, []) == list(data.dependents())


def test_column_names_can_be_separated_by_commas_or_spaces():
    from plottr.node.dataset_join import parseColumns

    assert parseColumns('Qi, Ql') == ['Qi', 'Ql']
    assert parseColumns('Qi Ql') == ['Qi', 'Ql']
    assert parseColumns('  Qi ,, Ql  ') == ['Qi', 'Ql']
    assert parseColumns('') == []


def test_the_open_dataset_is_named_after_its_file(qtbot, tmp_path):
    """`title` は読み込んだファイルのパス。凡例がそこから名前を取る。"""
    fc, node = _flowchart()
    here = _sweep(POWERS_A)
    here.add_meta('title', str(tmp_path / '2026-09-17T120000_ab-CD32_r1' / 'data.ddh5'))
    path = _write(tmp_path, 'CD32_r2', _sweep(POWERS_B))

    fc.setInput(dataIn=here)
    node.files = [path]
    names = set(fc.output()['dataOut'].dependents())

    assert 'Qi [CD32_r1]' in names and 'Qi [CD32_r2]' in names


def test_the_title_survives_the_join(qtbot, tmp_path):
    """図の上に出ている名前を落とさない。"""
    fc, node = _flowchart()
    here = _sweep(POWERS_A)
    here.add_meta('title', 'somewhere/2026-09-17T120000_ab-CD32_r1/data.ddh5')
    node.files = [_write(tmp_path, 'CD32_r2', _sweep(POWERS_B))]
    fc.setInput(dataIn=here)

    out = fc.output()['dataOut']
    assert out.meta_val('title') == here.meta_val('title')
