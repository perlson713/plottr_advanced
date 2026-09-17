"""Tests for the node that turns a dependent into the x axis.

The case this exists for: a resonator power sweep saves `photon` and `Qi` both
as dependents of `power`, so "Qi against the photon number" cannot be selected
in the GUI.  These tests check that the node produces exactly that dataset,
keeps the error bars attached, and refuses to invent data when the photon
number was never computed.

The other half is the line attenuation: `line_attenuation_db` is typed in by
hand, so it gets forgotten, and the photon number is NaN or wrong for a
measurement that is otherwise fine.  The node takes the number and has the
measurement repository recompute the column, so nothing has to be measured
again.
"""

import numpy as np
import pytest

from plottr.data.datadict import DataDict
from plottr.node.tools import linearFlowchart
from plottr.node.dependent_axis import DependentAsAxis

POWERS = [10.0, 0.0, -10.0, -20.0]
QI = [1.0e5, 1.4e5, 2.0e5, 2.6e5]
QC = [8.0e4, 8.0e4, 8.0e4, 8.0e4]
FR = 1.0e10
PHOTONS = [1.0e5, 1.0e4, 1.0e3, 1.0e2]


def _dataset(photons=None, expanded=False, points=5, parameters=True):
    """A dataset shaped like one of ours: a trace plus per-power fit results."""
    photons = PHOTONS if photons is None else photons
    f = np.linspace(9.99e9, 10.01e9, points)
    fields = dict(
        frequency=dict(unit='Hz'),
        power=dict(unit='dBm'),
        s11_raw=dict(axes=['frequency', 'power']),
        Qi=dict(axes=['power']),
        Qi_err=dict(axes=['power']),
        Ql=dict(axes=['power']),
        photon=dict(axes=['power']),
        fit_ok=dict(axes=['power']),
    )
    if parameters:
        # 光子数の計算し直しに要る列（測定スクリプトが保存しているもの）。
        fields['fr'] = dict(axes=['power'], unit='Hz')
        fields['Qc'] = dict(axes=['power'])
    data = DataDict(**fields)
    for power, qi, qc, photon in zip(POWERS, QI, QC, photons):
        extra = dict(fr=FR, Qc=qc) if parameters else {}
        data.add_data(frequency=f, power=power,
                      s11_raw=np.ones(f.size) * (1 + 1j),
                      Qi=qi, Qi_err=qi * 0.02, Ql=qi / 2.0,
                      photon=photon, fit_ok=1.0, **extra)
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


def test_the_values_are_left_alone_by_default(qtbot):
    """既定はそのままの光子数。対数にするのはプロット側の `Scale`（軸が
    10^4・10^5… と目盛られる）。ノードの `Logarithmic` は古いやり方で、
    **両方使うと 2 回対数を取ることになる**。"""
    fc, node = _flowchart()
    node.enabled = True
    fc.setInput(dataIn=_dataset())
    out = fc.output()['dataOut']

    assert out.axes() == ['photon']
    assert np.allclose(np.asarray(out.data_vals('photon')), np.sort(PHOTONS))


def test_log10_can_still_be_asked_for(qtbot):
    fc, node = _flowchart()
    node.enabled = True
    node.logAbscissa = True
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

    assert np.asarray(out.data_vals('photon')).size == len(POWERS)
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
    assert xy.xyAxes[0] == 'photon'
    assert out is not None and out.axes() == ['photon']


# ---------------------------------------------------------------------------
# ライン減衰量の入れ直し。
#
# `line_attenuation_db` は手で入れる値で、測ったトレースからは分からない。入れ
# 忘れると `photon` は全パワー NaN になり、別の配線の値のままだと桁ごとずれる。
# 光子数は保存済みの fr / Qc / Qi とパワーだけで決まるので、**測り直さずに**
# 直せる。計算は測定リポジトリ（`dataset_refit.recompute_photons`）が行う。
# ---------------------------------------------------------------------------


def _measurement_module():
    """測定リポジトリの `dataset_refit`。無ければテストを飛ばす。"""
    from plottr.node.resonator import resonator_fit_module

    module, message = resonator_fit_module()
    if module is None:
        pytest.skip(message)
    return module


def _expected_photons(attenuation):
    """このデータセットの正解（測定スクリプトと同じ式＝ resonator_tools）。"""
    module = _measurement_module()
    import resonator_fit as rf

    return [rf.photons_in_resonator(power - attenuation, fr=FR, Qc=qc, Qi=qi,
                                    port_type='reflection')
            for power, qc, qi in zip(POWERS, QC, QI)]


def test_attenuation_recomputes_a_photon_column_that_was_all_nan(qtbot):
    """入れ忘れて測ったデータでも、ここで入れれば photon が埋まる。"""
    _measurement_module()
    fc, node = _flowchart()
    node.attenuation = '70'
    fc.setInput(dataIn=_dataset(photons=[float('nan')] * len(POWERS)))
    out = fc.output()['dataOut']

    photons = np.asarray(out.data_vals('photon'), dtype=float).reshape(-1)
    assert np.all(np.isfinite(photons))
    assert np.allclose(photons, _expected_photons(70.0))


def test_attenuation_replaces_a_wrong_photon_column(qtbot):
    """別の配線の値のまま測っていた場合も、入れ直せば直る。"""
    _measurement_module()
    fc, node = _flowchart()
    node.attenuation = '60'
    fc.setInput(dataIn=_dataset())
    out = fc.output()['dataOut']

    photons = np.asarray(out.data_vals('photon'), dtype=float).reshape(-1)
    assert not np.allclose(photons, PHOTONS)
    assert np.allclose(photons, _expected_photons(60.0))
    # 減衰が 10 dB 増えれば、デバイスに入るパワーは 1/10 で光子数も 1/10。
    node.attenuation = '70'
    weaker = np.asarray(
        fc.output()['dataOut'].data_vals('photon'), dtype=float).reshape(-1)
    assert np.allclose(weaker * 10.0, photons)


def test_attenuation_works_without_the_axis_swap(qtbot):
    """入れ直しだけしたい（横軸はパワーのまま）場合。"""
    _measurement_module()
    fc, node = _flowchart()
    node.attenuation = '70'
    fc.setInput(dataIn=_dataset(photons=[float('nan')] * len(POWERS)))
    out = fc.output()['dataOut']

    assert out.axes() == ['frequency', 'power']   # 形は変わらない
    assert np.all(np.isfinite(
        np.asarray(out.data_vals('photon'), dtype=float)))
    # 何で計算したのかがデータセットに残る。
    assert np.allclose(
        np.asarray(out.data_vals('line_attenuation_db'), dtype=float), 70.0)


def test_attenuation_then_swap_gives_the_photon_axis(qtbot):
    _measurement_module()
    fc, node = _flowchart()
    node.attenuation = '70'
    node.logAbscissa = False
    node.enabled = True
    fc.setInput(dataIn=_dataset(photons=[float('nan')] * len(POWERS)))
    out = fc.output()['dataOut']

    assert out.axes() == ['photon']
    x = np.asarray(out.data_vals('photon'))
    assert x.size == len(POWERS) and np.all(np.diff(x) > 0)
    assert np.allclose(np.sort(_expected_photons(70.0)), x)


def test_port_type_reaches_the_recomputation(qtbot):
    """Not cosmetic: from resonator-tools 2.2.0 notch carries half the
    coefficient of reflection, so the wrong choice is wrong by a factor 2."""
    _measurement_module()
    import resonator_fit as rf

    fc, node = _flowchart()
    node.attenuation = '70'
    node.portType = 'notch'
    fc.setInput(dataIn=_dataset(photons=[float('nan')] * len(POWERS)))
    out = fc.output()['dataOut']

    photons = np.asarray(out.data_vals('photon'), dtype=float).reshape(-1)
    expected = [rf.photons_in_resonator(power - 70.0, fr=FR, Qc=qc, Qi=qi,
                                        port_type='notch')
                for power, qc, qi in zip(POWERS, QC, QI)]
    assert np.allclose(photons, expected)

    # 既定は reflection（この測定系はサーキュレータの先）。
    node.portType = 'reflection'
    again = np.asarray(
        fc.output()['dataOut'].data_vals('photon'), dtype=float).reshape(-1)
    assert np.allclose(again, _expected_photons(70.0))


def test_the_axis_label_carries_the_attenuation(qtbot):
    """10 dB は光子数のちょうど 1 桁ぶんで、plottr は自動スケールするので
    **曲線の形は変わらない**。軸の名前に入れておかないと、値を入れても何も
    起きていないように見える（実際にそう報告された）。"""
    _measurement_module()
    fc, node = _flowchart()
    node.enabled = True
    node.logAbscissa = True
    node.attenuation = '70'
    fc.setInput(dataIn=_dataset())
    out = fc.output()['dataOut']
    axis = out.axes()[0]
    assert '70' in out.label(axis)

    node.attenuation = '80'
    shifted = fc.output()['dataOut']
    assert '80' in shifted.label(axis)
    # 軸の名前は変えない。変えると下流の XYSelector が選択を失う。
    assert shifted.axes() == [axis]
    # 10 dB でちょうど 1 桁動く。
    assert np.allclose(np.asarray(out.data_vals(axis))
                       - np.asarray(shifted.data_vals(axis)), 1.0)

    # 減衰量を入れていなければ、ラベルはそのまま。
    node.attenuation = ''
    assert 'dB' not in fc.output()['dataOut'].label(axis)


def test_a_bad_attenuation_says_so_and_changes_nothing(qtbot):
    messages = []
    fc, node = _flowchart()
    node.statusChanged.connect(messages.append)
    node.attenuation = '70 dB'          # 単位まで書いてしまった
    fc.setInput(dataIn=_dataset())
    out = fc.output()['dataOut']

    assert np.allclose(
        np.asarray(out.data_vals('photon'), dtype=float).reshape(-1), PHOTONS)
    assert any('not a number' in m for m in messages)


def test_without_fit_parameters_it_explains_instead_of_failing(qtbot):
    """fr / Qc / Qi が無いデータセット（フィット前）でも落ちない。"""
    _measurement_module()
    messages = []
    fc, node = _flowchart()
    node.statusChanged.connect(messages.append)
    node.attenuation = '70'
    fc.setInput(dataIn=_dataset(parameters=False))
    out = fc.output()['dataOut']

    assert out is not None
    assert any('fr' in m and 'Qc' in m for m in messages)


def test_the_title_survives_the_swap(qtbot):
    """軸を入れ替えたデータセットにも、元の名前を残す。

    `title` は図の上に出る名前で、データセットを比べるときの見分けにも使う。
    作り直したデータセットで落とすと、その両方が消える（実際に消えていた）。
    """
    dataset = _dataset()
    dataset.add_meta('title', 'somewhere/2026-09-17T120000_ab-CD32_r1/data.ddh5')

    fc, node = _flowchart()
    node.enabled = True
    fc.setInput(dataIn=dataset)
    out = fc.output()['dataOut']

    assert out.meta_val('title') == dataset.meta_val('title')
