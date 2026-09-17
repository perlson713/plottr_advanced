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
