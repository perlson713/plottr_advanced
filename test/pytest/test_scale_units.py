from numpy.testing import assert_allclose

from plottr.data.datadict import DataDict
from plottr.node.tools import linearFlowchart
from plottr.node.scaleunits import ScaleUnits

import numpy as np


def test_basic_scale_units(qtbot):

    ScaleUnits.useUi = False
    ScaleUnits.uiClass = None

    fc = linearFlowchart(('scale_units', ScaleUnits))
    node = fc.nodes()['scale_units']

    x = np.arange(0, 5.0e-9, 1.0e-9)
    y = np.linspace(0, 1e9, 5)
    z = np.arange(4.0e6, 6.0e6, 1.0e6)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    vv = xx * yy * zz
    x1d, y1d, z1d = xx.flatten(), yy.flatten(), zz.flatten()
    v1d = vv.flatten()
    data = DataDict(
        x=dict(values=x1d, unit='V'),
        y=dict(values=y1d, unit="A"),
        z=dict(values=z1d, unit="Foobar"),
        vals=dict(values=v1d, axes=['x', 'y', 'z'])
    )
    assert data.validate()

    fc.setInput(dataIn=data)

    output = fc.outputValues()['dataOut']

    assert output['x']['unit'] == 'nV'
    assert_allclose(output['x']["values"],
                    (xx*1e9).ravel())

    assert output['y']['unit'] == 'GA'
    assert_allclose(output['y']["values"],
                    (yy / 1e9).ravel())

    assert output['z']["unit"] == '$10^{6}$ Foobar'
    assert_allclose(output['z']["values"],
                    (zz / 1e6).ravel())

    assert output['vals']['unit'] == ''
    assert_allclose(output['vals']['values'],
                    vv.flatten())


def _scaleUnitsFlowchart():
    ScaleUnits.useUi = False
    ScaleUnits.uiClass = None
    fc = linearFlowchart(('scale_units', ScaleUnits))
    return fc


def test_scale_units_hz_axis_to_ghz(qtbot):
    """A frequency sweep in Hz should be displayed in GHz."""
    fc = _scaleUnitsFlowchart()
    freq = np.linspace(1.2475e10, 1.2525e10, 11)
    data = DataDict(
        frequency=dict(values=freq, unit='Hz'),
        signal=dict(values=np.ones(freq.size), axes=['frequency']),
    )
    assert data.validate()
    fc.setInput(dataIn=data)
    out = fc.outputValues()['dataOut']

    assert out['frequency']['unit'] == 'GHz'
    assert_allclose(out['frequency']['values'], freq / 1e9)


def test_scale_units_leaves_unitless_data_untouched(qtbot):
    """The S11 case: unitless complex data and its error must not be rescaled."""
    fc = _scaleUnitsFlowchart()
    freq = np.linspace(1.2475e10, 1.2525e10, 11)
    s11 = 0.2 * np.exp(2j * np.pi * np.linspace(0, 1, freq.size))
    err = np.full(freq.size, 0.003)
    data = DataDict(
        frequency=dict(values=freq, unit='Hz'),
        S11=dict(values=s11, axes=['frequency'], unit=''),
        S11_error=dict(values=err, axes=['frequency'], unit=''),
    )
    data.add_meta('errorbar', 'S11_error', data='S11')
    assert data.validate()
    fc.setInput(dataIn=data)
    out = fc.outputValues()['dataOut']

    assert out['S11']['unit'] == ''
    assert out['S11_error']['unit'] == ''
    assert_allclose(out['S11']['values'], s11)
    assert_allclose(out['S11_error']['values'], err)


def test_scale_units_error_column_shares_parent_scale(qtbot):
    """An error column must never get a different prefix than its dependent.

    ``errorBarData`` only compares shapes, so a mismatch would silently plot
    error bars that are orders of magnitude wrong.
    """
    fc = _scaleUnitsFlowchart()
    x = np.arange(5.0)
    sig = np.linspace(1e-4, 1e-3, 5)      # -> milli
    err = np.full(5, 3e-6)                # -> micro, if scaled on its own
    data = DataDict(
        x=dict(values=x, unit=''),
        signal=dict(values=sig, axes=['x'], unit='V'),
        signal_error=dict(values=err, axes=['x'], unit='V'),
    )
    data.add_meta('errorbar', 'signal_error', data='signal')
    assert data.validate()
    fc.setInput(dataIn=data)
    out = fc.outputValues()['dataOut']

    assert out['signal']['unit'] == out['signal_error']['unit']
    # the ratio between value and error must be preserved
    assert_allclose(out['signal_error']['values'] / out['signal']['values'],
                    err / sig)


def test_scale_units_mismatched_units_are_scaled_independently(qtbot):
    """If the error has a different unit, scale it on its own (and don't crash)."""
    fc = _scaleUnitsFlowchart()
    x = np.arange(5.0)
    sig = np.linspace(1e-4, 1e-3, 5)
    err = np.full(5, 1.0)
    data = DataDict(
        x=dict(values=x, unit=''),
        signal=dict(values=sig, axes=['x'], unit='V'),
        signal_error=dict(values=err, axes=['x'], unit='%'),
    )
    data.add_meta('errorbar', 'signal_error', data='signal')
    assert data.validate()
    fc.setInput(dataIn=data)
    out = fc.outputValues()['dataOut']

    assert out['signal']['unit'] == 'mV'
    assert out['signal_error']['unit'].endswith('%')
