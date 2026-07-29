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
    ERROR_BAR_KEY,
    ERROR_BAR_X_KEY,
    DB_PER_NEPER,
    toDecibel,
)


def _complexPlotItem(x, z, sigma=None):
    """Helper: build a PlotItem for 1d complex data, optionally with errors."""
    plotOptions = {} if sigma is None else {ERROR_BAR_KEY: np.asarray(sigma)}
    return PlotItem(
        data=[np.asarray(x), np.asarray(z)],
        id=0,
        subPlot=0,
        plotDataType=PlotDataType.line1d,
        labels=['x', 'S'],
        plotOptions=plotOptions,
    )


def _split(rep, x, z, sigma=None, **fmAttrs):
    """Helper: run _splitComplexData for a given complex representation."""
    fm = AutoFigureMaker()
    fm.complexRepresentation = rep
    for k, v in fmAttrs.items():
        setattr(fm, k, v)
    return fm._splitComplexData(_complexPlotItem(x, z, sigma))


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


def test_log_mag_and_phase_applies_db():
    """logMag must actually be 20*log10, not linear magnitude nor 20*ln."""
    x = np.arange(3.0)
    z = np.array([1.0 + 0j, 0.1 + 0j, 0.01 + 0j])

    mag_item, phase_item = _split(ComplexRepresentation.log_MagAndPhase, x, z)

    assert np.allclose(np.asarray(mag_item.data[-1]), [0.0, -20.0, -40.0])
    assert mag_item.labels[-1].endswith('(dB)')
    # guards against a re-introduced natural-log transform (would give -46.05)
    assert not np.allclose(np.asarray(mag_item.data[-1]),
                           20 * np.log(np.abs(z)))


def test_to_decibel_masks_non_positive_magnitude():
    """|z| = 0 has no dB value; it must be masked, not -inf, and not raise."""
    z = np.array([1.0 + 0j, 0.0 + 0j, 0.5 + 0j])
    db = toDecibel(z)
    assert np.ma.is_masked(db[1])
    assert np.isfinite(db[0]) and np.isfinite(db[2])

    # also through the full split, and for a masked input array
    zm = np.ma.masked_array(z, mask=[False, False, False])
    mag_item, _ = _split(ComplexRepresentation.log_MagAndPhase, np.arange(3.0), zm)
    values = np.asarray(mag_item.data[-1])
    assert not np.any(np.isinf(values[~np.ma.getmaskarray(mag_item.data[-1])]))


def test_mag_phase_error_propagation():
    """Magnitude keeps sigma; phase error is sigma/|z| (radians)."""
    x = np.arange(3.0)
    z = np.array([1.0 + 0j, 0.1 + 0j, 0.5 + 0.5j])
    sigma = np.full(3, 0.003)

    mag_item, phase_item = _split(ComplexRepresentation.magAndPhase, x, z, sigma)

    assert np.allclose(np.asarray(mag_item.plotOptions[ERROR_BAR_KEY]), sigma)
    assert np.allclose(np.asarray(phase_item.plotOptions[ERROR_BAR_KEY]),
                       sigma / np.abs(z))
    assert 'rad' in phase_item.labels[-1]


def test_db_error_propagation():
    """dB error is 20/ln(10) * sigma/|z|."""
    x = np.arange(3.0)
    z = np.array([1.0 + 0j, 0.1 + 0j, 0.5 + 0.5j])
    sigma = np.full(3, 0.003)

    mag_item, _ = _split(ComplexRepresentation.log_MagAndPhase, x, z, sigma)

    assert np.allclose(np.asarray(mag_item.plotOptions[ERROR_BAR_KEY]),
                       DB_PER_NEPER * sigma / np.abs(z))


def test_phase_in_degrees_converts_values_and_errors():
    x = np.arange(3.0)
    z = np.array([1.0 + 0j, 1.0 + 1.0j, 0.5 + 0j])
    sigma = np.full(3, 0.003)

    _, phase_item = _split(ComplexRepresentation.magAndPhase, x, z, sigma,
                           phaseDegrees=True)

    assert np.allclose(np.asarray(phase_item.data[-1]),
                       np.degrees(np.angle(z)))
    assert np.allclose(np.asarray(phase_item.plotOptions[ERROR_BAR_KEY]),
                       sigma / np.abs(z) * 180.0 / np.pi)
    assert 'deg' in phase_item.labels[-1]


def test_phase_unwrap_toggle():
    x = np.linspace(0, 1, 200)
    z = np.exp(1j * np.linspace(0, 6 * np.pi, 200))

    _, wrapped = _split(ComplexRepresentation.magAndPhase, x, z)
    _, unwrapped = _split(ComplexRepresentation.magAndPhase, x, z,
                          phaseUnwrap=True)

    w = np.asarray(wrapped.data[-1])
    u = np.asarray(unwrapped.data[-1])
    assert w.min() >= -np.pi - 1e-9 and w.max() <= np.pi + 1e-9
    assert np.all(np.diff(u) > 0)          # monotonic once unwrapped
    assert np.isclose(np.ptp(u), 6 * np.pi, rtol=1e-2)


def test_complex_plane_error_bars_are_radial():
    """An isotropic uncertainty must appear on BOTH axes, not just y."""
    x = np.arange(3.0)
    z = np.array([1.0 + 0j, 0.1 + 0j, 0.5 + 0.5j])
    sigma = np.full(3, 0.003)

    item, = _split(ComplexRepresentation.complexPlane, x, z, sigma)

    assert np.allclose(np.asarray(item.plotOptions[ERROR_BAR_KEY]), sigma)
    assert np.allclose(np.asarray(item.plotOptions[ERROR_BAR_X_KEY]), sigma)


def test_real_and_imag_error_bars_are_independent_arrays():
    x = np.arange(3.0)
    z = np.array([1.0 + 1j, 0.1 + 0j, 0.5 + 0.5j])
    sigma = np.full(3, 0.003)

    re_item, im_item = _split(ComplexRepresentation.realAndImag, x, z, sigma)

    re_err = re_item.plotOptions[ERROR_BAR_KEY]
    im_err = im_item.plotOptions[ERROR_BAR_KEY]
    assert np.allclose(np.asarray(re_err), sigma)
    assert np.allclose(np.asarray(im_err), sigma)

    re_err[0] = 99.0                        # must not leak into the other half
    assert not np.isclose(np.asarray(im_err)[0], 99.0)


def test_split_without_errors_adds_no_private_keys():
    """Items without error bars must not gain private keys.

    plotLine splats plotOptions into Axes.plot, which rejects unknown kwargs.
    """
    x = np.arange(3.0)
    z = np.array([1.0 + 0j, 0.1 + 0j, 0.5 + 0.5j])

    for rep in ComplexRepresentation:
        for item in _split(rep, x, z):
            assert ERROR_BAR_KEY not in item.plotOptions
            assert ERROR_BAR_X_KEY not in item.plotOptions


def test_plot_line_draws_x_and_y_error_bars():
    """The mpl backend must consume xerr, and must pop both private keys."""
    from plottr.plot.mpl.autoplot import FigureMaker

    fig = plt.figure()
    x = np.linspace(0, 1, 20)
    z = np.exp(2j * np.pi * x)
    with FigureMaker(fig) as fm:
        fm.plotType = PlotType.singletraces
        fm.complexRepresentation = ComplexRepresentation.complexPlane
        fm.addData(x, z, labels=['x', 'S'], plotDataType=PlotDataType.line1d,
                   **{ERROR_BAR_KEY: np.full(x.size, 0.05)})

    ax = fig.axes[0]
    assert len(ax.containers) == 1
    assert ax.containers[0].has_xerr
    assert ax.containers[0].has_yerr
    plt.close(fig)


def test_error_bar_helpers_are_reexported_from_plot_base():
    """plottr.plot.base must keep re-exporting the moved data-layer helpers."""
    from plottr.data import datadict as dd
    import plottr.plot.base as pb

    assert pb.errorBarData is dd.errorBarData
    assert pb.errorBarDataName is dd.errorBarDataName
    assert pb.plottableDependents is dd.plottableDependents


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
