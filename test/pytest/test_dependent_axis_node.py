"""Tests for the node that turns a dependent into the x axis.

The case this exists for: a resonator power sweep saves `photon` and `Qi` both
as dependents of `power`, so "Qi against the photon number" cannot be selected
in the GUI.  These tests check that the node produces exactly that dataset,
keeps the error bars attached, and refuses to invent data when the photon
number was never computed.
"""

import numpy as np
import pytest

from plottr.data.datadict import DataDict
from plottr.node.tools import linearFlowchart
from plottr.node.dependent_axis import DependentAsAxis

POWERS = [10.0, 0.0, -10.0, -20.0]
QI = [1.0e5, 1.4e5, 2.0e5, 2.6e5]
PHOTONS = [1.0e5, 1.0e4, 1.0e3, 1.0e2]


def _dataset(photons=None, expanded=False, points=5):
    """A dataset shaped like one of ours: a trace plus per-power fit results."""
    photons = PHOTONS if photons is None else photons
    f = np.linspace(9.99e9, 10.01e9, points)
    data = DataDict(
        frequency=dict(unit='Hz'),
        power=dict(unit='dBm'),
        s11_raw=dict(axes=['frequency', 'power']),
        Qi=dict(axes=['power']),
        Qi_err=dict(axes=['power']),
        Ql=dict(axes=['power']),
        photon=dict(axes=['power']),
        fit_ok=dict(axes=['power']),
    )
    for power, qi, photon in zip(POWERS, QI, photons):
        data.add_data(frequency=f, power=power,
                      s11_raw=np.ones(f.size) * (1 + 1j),
                      Qi=qi, Qi_err=qi * 0.02, Ql=qi / 2.0,
                      photon=photon, fit_ok=1.0)
    data.validate()
    return data.expand() if expanded else data


def _flowchart():
    DependentAsAxis.useUi = False
    fc = linearFlowchart(('swap', DependentAsAxis))
    return fc, fc.nodes()['swap']


def test_disabled_node_passes_data_through(qtbot):
    dataset = _dataset()
    fc, node = _flowchart()
    fc.setInput(dataIn=dataset)
    out = fc.output()['dataOut']
    assert out.axes() == ['frequency', 'power']
    assert 's11_raw' in out.dependents()


@pytest.mark.parametrize('expanded', [False, True])
def test_photon_becomes_the_axis(qtbot, expanded):
    """Qi against the photon number, one point per power, ascending."""
    fc, node = _flowchart()
    node.logAbscissa = False
    node.enabled = True
    fc.setInput(dataIn=_dataset(expanded=expanded))
    out = fc.output()['dataOut']

    assert out.axes() == ['photon']
    assert out.axes('Qi') == ['photon']

    x = np.asarray(out.data_vals('photon'))
    assert x.size == len(POWERS)          # 展開されていても 1 パワー 1 点
    assert np.all(np.diff(x) > 0)         # x 昇順（線が行ったり来たりしない）

    # 値の対応が保たれている
    order = np.argsort(PHOTONS)
    assert np.allclose(x, np.asarray(PHOTONS)[order])
    assert np.allclose(np.asarray(out.data_vals('Qi')), np.asarray(QI)[order])


def test_trace_dependents_are_dropped(qtbot):
    """s11_raw lives on frequency too, so it cannot be drawn against photon."""
    fc, node = _flowchart()
    node.enabled = True
    fc.setInput(dataIn=_dataset())
    out = fc.output()['dataOut']
    assert 's11_raw' not in out.dependents()
    assert {'Qi', 'Ql', 'fit_ok'}.issubset(set(out.dependents()))


def test_error_bars_travel_with_their_partner(qtbot):
    """`<dep>_err` keeps resolving as the error bar of `<dep>`."""
    from plottr.plot.base import errorBarDataName

    fc, node = _flowchart()
    node.enabled = True
    fc.setInput(dataIn=_dataset())
    out = fc.output()['dataOut']
    assert errorBarDataName(out, 'Qi') == 'Qi_err'


def test_log10_is_the_default(qtbot):
    fc, node = _flowchart()
    node.enabled = True
    fc.setInput(dataIn=_dataset())
    out = fc.output()['dataOut']

    assert out.axes() == ['log10_photon']
    assert np.allclose(np.asarray(out.data_vals('log10_photon')),
                       np.sort(np.log10(PHOTONS)))


def test_all_nan_photon_says_so_and_keeps_the_data(qtbot):
    """line_attenuation_db unset -> photon is NaN everywhere."""
    messages = []
    fc, node = _flowchart()
    node.statusChanged.connect(messages.append)
    node.enabled = True
    fc.setInput(dataIn=_dataset(photons=[float('nan')] * len(POWERS)))
    out = fc.output()['dataOut']

    assert out.axes() == ['frequency', 'power']       # 元のまま返す
    assert any('line_attenuation_db' in m for m in messages)


def test_points_without_a_photon_number_are_dropped(qtbot):
    photons = [1.0e5, float('nan'), 1.0e3, 1.0e2]
    fc, node = _flowchart()
    node.logAbscissa = False
    node.enabled = True
    fc.setInput(dataIn=_dataset(photons=photons))
    out = fc.output()['dataOut']

    x = np.asarray(out.data_vals('photon'))
    assert x.size == 3
    assert np.all(np.isfinite(x))
    # 落ちたのは 2 番目のパワーだけ
    assert np.allclose(np.asarray(out.data_vals('Qi')),
                       np.asarray([QI[3], QI[2], QI[0]]))


def test_failed_fits_stay_as_gaps(qtbot):
    """A NaN Qi is a real gap in the sweep; it must not be silently removed."""
    dataset = _dataset()
    dataset['Qi']['values'] = np.asarray(dataset['Qi']['values'], dtype=float)
    dataset['Qi']['values'][1] = np.nan

    fc, node = _flowchart()
    node.enabled = True
    fc.setInput(dataIn=dataset)
    out = fc.output()['dataOut']

    assert np.asarray(out.data_vals('log10_photon')).size == len(POWERS)
    assert np.isnan(np.asarray(out.data_vals('Qi'))).sum() == 1


def test_downstream_x_axis_follows_the_swap(qtbot):
    """The plot must not keep showing the old axis.

    `XYSelector` remembers which axis is x.  When this node replaces the axes,
    that name is gone; refusing to produce output would leave the plot showing
    the previous dataset, which looks exactly like "the node did nothing".
    """
    from plottr.node.dim_reducer import XYSelector

    XYSelector.useUi = False
    fc = linearFlowchart(('swap', DependentAsAxis), ('xy', XYSelector))
    node, xy = fc.nodes()['swap'], fc.nodes()['xy']

    # power だけを軸に持つ、そのまま XYSelector に渡せる形のデータ。
    simple = DataDict(
        power=dict(unit='dBm'),
        Qi=dict(axes=['power']),
        photon=dict(axes=['power']),
    )
    for power, qi, photon in zip(POWERS, QI, PHOTONS):
        simple.add_data(power=power, Qi=qi, photon=photon)
    simple.validate()

    fc.setInput(dataIn=simple)
    xy.xyAxes = ('power', None)
    assert fc.output()['dataOut'].axes() == ['power']

    node.enabled = True
    out = fc.output()['dataOut']
    assert xy.xyAxes[0] == 'log10_photon'
    assert out is not None and out.axes() == ['log10_photon']
