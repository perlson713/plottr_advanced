import numpy as np

# from plottr.apps.tools import make_sequential_flowchart
from plottr.node.tools import linearFlowchart
from plottr.node.data_selector import DataSelector
from plottr.utils import testdata


def test_data_extraction(qtbot):
    """
    Test whether extraction of one dependent gives the right data back.
    """
    DataSelector.useUi = False

    data = testdata.three_compatible_3d_sets()
    data_name = data.dependents()[0]
    field_names = [data_name] + data.axes(data_name)

    fc = linearFlowchart(('selector', DataSelector))
    node = fc.nodes()['selector']

    fc.setInput(dataIn=data)
    node.selectedData = data_name
    out = fc.output()['dataOut']

    assert out.dependents() == [data_name]
    for d, _ in out.data_items():
        assert d in field_names

    assert np.all(np.isclose(
        data.data_vals(data_name), out.data_vals(data_name),
        atol=1e-15
    ))


def test_data_extraction2(qtbot):
    """
    Test whether extraction of two dependents gives the right data back.
    """
    DataSelector.useUi = False

    data = testdata.three_compatible_3d_sets()
    data_names = [data.dependents()[0], data.dependents()[1]]
    field_names = data_names + data.axes(data_names[0])

    fc = linearFlowchart(('selector', DataSelector))
    node = fc.nodes()['selector']

    fc.setInput(dataIn=data)
    node.selectedData = data_names
    out = fc.output()['dataOut']

    assert out.dependents() == data_names
    for d, _ in out.data_items():
        assert d in field_names

    assert np.all(np.isclose(
        data.data_vals(data_names[0]), out.data_vals(data_names[0]),
        atol=1e-15
    ))

    assert np.all(np.isclose(
        data.data_vals(data_names[1]), out.data_vals(data_names[1]),
        atol=1e-15
    ))


def test_incompatible_sets(qtbot):
    """
    Test that selecting incompatible data sets give None output.
    """
    DataSelector.useUi = False

    data = testdata.three_incompatible_3d_sets()
    fc = linearFlowchart(('selector', DataSelector))
    node = fc.nodes()['selector']
    fc.setInput(dataIn=data)
    node.selectedData = data.dependents()[0], data.dependents()[1]
    assert fc.output()['dataOut'] == None

    node.selectedData = data.dependents()[0]
    assert fc.output()['dataOut'].dependents() == [data.dependents()[0]]

    node.selectedData = data.dependents()[1]
    assert fc.output()['dataOut'].dependents() == [data.dependents()[1]]


def _errorBarDataDict(errorFieldName='signal_std', useMeta=True,
                      errorAxes=None):
    """Build a DataDict with a dependent that has an error-bar column."""
    from plottr.data.datadict import DataDict

    x = np.arange(5.0)
    fields = dict(
        x=dict(values=x),
        signal=dict(values=np.linspace(0, 1, 5), axes=['x']),
    )
    if errorAxes is None:
        errorAxes = ['x']
        errorValues = np.full(5, 0.1)
    else:
        # incompatible: lives on a different axis
        fields['y'] = dict(values=np.arange(5.0))
        errorValues = np.full(5, 0.1)
    fields[errorFieldName] = dict(values=errorValues, axes=errorAxes)

    data = DataDict(**fields)
    if useMeta:
        data.add_meta('errorbar', errorFieldName, data='signal')
    assert data.validate()
    return data


def test_data_selector_keeps_meta_error_column(qtbot):
    """Selecting only the dependent must not discard its error-bar column."""
    from plottr.data.datadict import errorBarDataName, plottableDependents

    DataSelector.useUi = False
    data = _errorBarDataDict()

    fc = linearFlowchart(('selector', DataSelector))
    node = fc.nodes()['selector']
    fc.setInput(dataIn=data)
    node.selectedData = ['signal']
    out = fc.output()['dataOut']

    assert 'signal_std' in out
    assert errorBarDataName(out, 'signal') == 'signal_std'
    assert plottableDependents(out) == ['signal']
    # the user's selection itself must be left alone
    assert node.selectedData == ['signal']


def test_data_selector_keeps_name_convention_error_column(qtbot):
    """The naming-convention fallback (signal_err) must work too."""
    from plottr.data.datadict import errorBarDataName

    DataSelector.useUi = False
    data = _errorBarDataDict(errorFieldName='signal_err', useMeta=False)

    fc = linearFlowchart(('selector', DataSelector))
    node = fc.nodes()['selector']
    fc.setInput(dataIn=data)
    node.selectedData = ['signal']
    out = fc.output()['dataOut']

    assert errorBarDataName(out, 'signal') == 'signal_err'


def test_data_selector_skips_incompatible_error_column(qtbot):
    """An error field on different axes must not be dragged in."""
    DataSelector.useUi = False
    data = _errorBarDataDict(errorAxes=['y'])

    fc = linearFlowchart(('selector', DataSelector))
    node = fc.nodes()['selector']
    fc.setInput(dataIn=data)
    node.selectedData = ['signal']
    out = fc.output()['dataOut']

    assert out is not None
    assert 'signal_std' not in out
    assert out.dependents() == ['signal']


def test_data_selector_include_error_bars_can_be_disabled(qtbot):
    """The escape hatch restores the old behaviour."""
    from plottr.data.datadict import errorBarDataName

    DataSelector.useUi = False
    data = _errorBarDataDict()

    fc = linearFlowchart(('selector', DataSelector))
    node = fc.nodes()['selector']
    node.include_error_bars = False
    fc.setInput(dataIn=data)
    node.selectedData = ['signal']
    out = fc.output()['dataOut']

    assert 'signal_std' not in out
    assert errorBarDataName(out, 'signal') is None
