"""Tests for the resonator fit node.

The physics lives in ``resonator_fit.py`` of the measurement repository, so
these tests are skipped when that module (and ``resonator_tools``, which it
needs) is not importable.  What is tested here is the node's part of the job:
finding the trace, splitting the dataset into one trace per power, and putting
the fitted curve back on the same axes so it can be plotted on top of the data.
"""

import math

import numpy as np
import pytest

from plottr.data.datadict import DataDict
from plottr.node.tools import linearFlowchart
from plottr.node.resonator import ResonatorFit, resonator_fit_module

refit, _message = resonator_fit_module()
pytestmark = pytest.mark.skipif(
    refit is None, reason='dataset_refit.py is not importable (RESONATOR_FIT_PATH)')
rf = getattr(refit, 'rf', None)

FR = 10e9
QL = 5000.0
QC = QL / 0.9  # strongly overcoupled reflection, as in the lab
POWERS = [-10.0, -20.0]


def _dataset(expanded: bool = False, noise: float = 5e-4) -> DataDict:
    """A dataset shaped like one of ours, without any fit columns in it."""
    fwhm = FR / QL
    f = np.linspace(FR - 8 * fwhm, FR + 8 * fwhm, 401)
    environment = 0.35 * np.exp(0.9j) * np.exp(-2j * np.pi * f * 2e-9)
    rng = np.random.default_rng(0)

    data = DataDict(
        frequency=dict(unit='Hz'),
        power=dict(unit='dBm'),
        s11=dict(axes=['frequency', 'power']),
    )
    for power in POWERS:
        trace = environment * rf.reflection_model(f, FR, QL, QC, a=0.0)
        trace = trace + (rng.standard_normal(f.size)
                         + 1j * rng.standard_normal(f.size)) * noise
        data.add_data(frequency=f, power=power, s11=trace)
    data.validate()
    return data.expand() if expanded else data


def _flowchart():
    ResonatorFit.useUi = False
    fc = linearFlowchart(('fit', ResonatorFit))
    node = fc.nodes()['fit']
    node.portType = 'reflection'
    return fc, node


def test_disabled_node_passes_data_through(qtbot):
    dataset = _dataset()
    fc, node = _flowchart()
    fc.setInput(dataIn=dataset)
    assert fc.outputValues()['dataOut'] is dataset
    assert node.enabled is False  # off unless asked for: fitting takes time


@pytest.mark.parametrize('expanded', [False, True])
def test_fit_curve_is_added_on_the_same_axes(qtbot, expanded):
    dataset = _dataset(expanded=expanded)
    fc, node = _flowchart()
    fc.setInput(dataIn=dataset)
    node.enabled = True

    out = fc.outputValues()['dataOut']
    assert out is not None
    for name in ('s11_fit', 's11_cor', 's11_cor_fit'):
        assert name in out.dependents()
        assert out.axes(name) == out.axes('s11')
        assert out.data_vals(name).shape == out.data_vals('s11').shape

    # 重ね描きの本体: フィット曲線が生データに乗っていること。
    data = out.data_vals('s11')
    fit = out.data_vals('s11_fit')
    assert np.all(np.isfinite(fit))
    residual = float(np.sqrt(np.mean(np.abs(data - fit) ** 2)))
    assert residual < 0.05 * float(np.mean(np.abs(data)))

    # フィットパラメータはパワー軸に載る。
    assert out.axes('s11_fit_Ql') == ['power']
    fitted = np.unique(out.data_vals('s11_fit_Ql'))
    assert len(fitted) == len(POWERS)
    assert np.all(np.abs(fitted - QL) / QL < 0.1)


def test_traces_are_fitted_one_power_at_a_time(qtbot):
    """Each power must be fitted on its own; mixing them up would show up as
    a fit that is right for neither."""
    dataset = _dataset()
    fc, node = _flowchart()
    fc.setInput(dataIn=dataset)
    node.enabled = True
    out = fc.outputValues()['dataOut']

    powers = out.data_vals('power')
    fit = out.data_vals('s11_fit')
    data = out.data_vals('s11')
    for power in POWERS:
        mask = powers == power
        assert mask.sum() > 8
        residual = float(np.sqrt(np.mean(np.abs(data[mask] - fit[mask]) ** 2)))
        assert residual < 0.05 * float(np.mean(np.abs(data[mask])))


def test_nothing_to_fit_is_not_an_error(qtbot):
    data = DataDict(x=dict(), y=dict(values=np.arange(10.0), axes=['x']))
    data['x']['values'] = np.arange(10.0)
    data.validate()

    fc, node = _flowchart()
    fc.setInput(dataIn=data)
    node.enabled = True
    out = fc.outputValues()['dataOut']
    assert out.dependents() == ['y']


def test_the_fit_is_cached_between_runs(qtbot):
    dataset = _dataset()
    fc, node = _flowchart()
    fc.setInput(dataIn=dataset)
    node.enabled = True
    first = fc.outputValues()['dataOut']

    # 同じデータ・同じ設定なら当て直さない（キャッシュがそのまま返る）。
    node.update()
    assert fc.outputValues()['dataOut'] is first

    # 設定が変われば当て直す。
    node.portType = 'notch'
    assert fc.outputValues()['dataOut'] is not first


def test_saved_fit_columns_are_not_clobbered(qtbot):
    """A dataset that already carries a fit (the measurement script saves one)
    must keep it; the node's own curve goes next to it."""
    dataset = _dataset()
    dataset['s11_fit'] = dict(values=np.zeros_like(dataset.data_vals('s11')),
                              axes=['frequency', 'power'])
    dataset['s11_cor'] = dict(values=np.zeros_like(dataset.data_vals('s11')),
                              axes=['frequency', 'power'])
    dataset.validate()

    fc, node = _flowchart()
    fc.setInput(dataIn=dataset)
    node.enabled = True
    out = fc.outputValues()['dataOut']

    assert np.all(out.data_vals('s11_fit') == 0)  # 保存済みの列はそのまま
    assert 's11_fit_refit' in out.dependents()
    assert np.all(np.isfinite(out.data_vals('s11_fit_refit')))
    assert 's11_fit_refit_Ql' in out.dependents()


def test_the_measured_trace_is_preferred_over_a_calibrated_copy():
    # `s11_cor` は較正済みのコピー。既定で当てるのは測定そのままのトレース。
    assert refit.default_dependent(['s11_cor', 's11_raw']) == 's11_raw'
    assert refit.default_dependent(['s21_normalized', 's21']) == 's21'
    assert refit.default_dependent(['s11_cor']) == 's11_cor'
    assert refit.default_dependent([]) == ''
