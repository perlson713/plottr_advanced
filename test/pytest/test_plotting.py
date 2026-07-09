import matplotlib.pyplot as plt
import numpy as np
from plottr.plot.mpl.plotting import PlotType, colorplot2d
from plottr.data.datadict import DataDict
from plottr.plot.base import (
    AutoFigureMaker,
    ComplexRepresentation,
    PlotDataType,
    PlotItem,
    errorBarData,
    errorBarDataName,
    plottableDependents,
)


def test_colorplot2d_scatter_rgba_error():
    """
    Check that scatter plots are not trying to plot 1x3 and 1x4
    z arrays as rgb(a) colors.

    """
    fig, ax = plt.subplots(1, 1)
    x = np.array([[0.0, 11.11111111, 22.22222222, 33.33333333]])
    y = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )
    z = np.array([[5.08907021, 4.93923391, 5.11400073, 5.0925613]])
    colorplot2d(ax, x, y, z, PlotType.scatter2d)

    x = np.array([[0.0, 11.11111111, 22.22222222]])
    y = np.array([[0.0, 0.0, 0.0]])
    z = np.array([[5.08907021, 4.93923391, 5.11400073]])
    colorplot2d(ax, x, y, z, PlotType.scatter2d)


def test_errorbar_column_from_dependent_metadata():
    dd = DataDict(
        x=dict(values=np.arange(3)),
        signal=dict(values=np.arange(3.0), axes=['x']),
        signal_std=dict(values=np.ones(3) * 0.1, axes=['x']),
    )
    dd.add_meta('errorbar', 'signal_std', data='signal')

    assert errorBarDataName(dd, 'signal') == 'signal_std'
    assert np.allclose(errorBarData(dd, 'signal'), np.ones(3) * 0.1)
    assert plottableDependents(dd) == ['signal']


def test_errorbar_column_from_name_convention():
    dd = DataDict(
        x=dict(values=np.arange(3)),
        signal=dict(values=np.arange(3.0), axes=['x']),
        signal_err=dict(values=np.ones(3) * 0.2, axes=['x']),
    )

    assert errorBarDataName(dd, 'signal') == 'signal_err'
    assert np.allclose(errorBarData(dd, 'signal'), np.ones(3) * 0.2)
    assert plottableDependents(dd) == ['signal']


def test_complex_plane_representation_splits_to_real_vs_imag():
    fm = AutoFigureMaker()
    fm.complexRepresentation = ComplexRepresentation.complexPlane
    plot_item = PlotItem(
        data=[np.arange(3), np.array([1 + 2j, 3 + 4j, 5 + 6j])],
        id=0,
        subPlot=0,
        plotDataType=PlotDataType.scatter1d,
        labels=['x', 'signal'],
        plotOptions={},
    )

    items = fm._splitComplexData(plot_item)

    assert len(items) == 1
    assert items[0].plotDataType is PlotDataType.scatter1d
    assert np.allclose(items[0].data[0], [1, 3, 5])
    assert np.allclose(items[0].data[1], [2, 4, 6])
    assert items[0].labels == ['Real(signal)', 'Imag(signal)']
