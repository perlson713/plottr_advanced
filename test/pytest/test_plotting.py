import matplotlib.pyplot as plt
import numpy as np
from plottr.plot.mpl.plotting import (
    PlotType,
    colorplot2d,
    AxesOptions,
    square_axes,
    apply_axes_options,
    apply_axes_options_to_figure,
    data_axes,
)
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


def test_square_axes_makes_range_square_and_equal_aspect():
    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])  # very wide range in x, narrow in y
    square_axes(ax)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # spans are equal
    assert np.isclose(x1 - x0, y1 - y0)
    # aspect is equal (data coords are square)
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_square_axes_handles_degenerate_range():
    fig, ax = plt.subplots()
    ax.plot([5], [5])
    ax.set_xlim(5, 5)
    ax.set_ylim(5, 5)
    square_axes(ax)
    x0, x1 = ax.get_xlim()
    assert x1 > x0  # no zero-width axes
    plt.close(fig)


def test_axes_options_default_is_noop():
    options = AxesOptions()
    assert options.isDefault()

    fig, ax = plt.subplots()
    ax.set_xscale('log')  # some pre-existing state
    apply_axes_options(ax, options)
    assert ax.get_xscale() == 'log'  # untouched
    plt.close(fig)


def test_axes_options_apply_scale_grid_aspect_and_limits():
    options = AxesOptions(
        xscale='log', yscale='linear',
        xmin=1.0, xmax=100.0, ymin=-2.0, ymax=2.0,
        grid=True, equalAspect=True,
    )
    assert not options.isDefault()

    fig, ax = plt.subplots()
    ax.plot([1, 10, 100], [0, 1, 2])
    apply_axes_options(ax, options)

    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'linear'
    assert np.allclose(ax.get_xlim(), (1.0, 100.0))
    assert np.allclose(ax.get_ylim(), (-2.0, 2.0))
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_axes_options_partial_limit_keeps_other_bound():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlim(0.0, 1.0)
    apply_axes_options(ax, AxesOptions(xmax=5.0))
    x0, x1 = ax.get_xlim()
    assert np.isclose(x0, 0.0)  # lower bound preserved
    assert np.isclose(x1, 5.0)  # upper bound applied
    plt.close(fig)


def test_apply_axes_options_to_figure_skips_colorbar():
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.rand(4, 4))
    fig.colorbar(im, ax=ax)

    assert len(data_axes(fig)) == 1  # colorbar excluded

    apply_axes_options_to_figure(fig, AxesOptions(grid=True, equalAspect=True))
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_complex_plane_figuremaker_produces_square_axes():
    from plottr.plot.mpl.autoplot import FigureMaker

    fig = plt.figure()
    with FigureMaker(fig) as fm:
        fm.plotType = PlotType.singletraces
        fm.complexRepresentation = ComplexRepresentation.complexPlane
        x = np.linspace(0, 1, 50)
        z = np.exp(2j * np.pi * x)  # unit circle in the complex plane
        fm.addData(x, z, labels=['x', 'signal'],
                   plotDataType=PlotDataType.line1d)

    ax = fig.axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert np.isclose(x1 - x0, y1 - y0)
    assert ax.get_aspect() == 1.0
    plt.close(fig)
