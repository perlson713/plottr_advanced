"""
``plottr.plot.mpl.plotting`` -- Plotting tools (mostly used in Autoplot)
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Any, Dict, List, Optional, Tuple, Union, cast, \
    OrderedDict as OrderedDictType

import numpy as np
from matplotlib import colors, rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter

from plottr.utils import num
from plottr.utils.num import centers2edges_2d, interp_meshgrid_2d

__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'


@unique
class PlotType(Enum):
    """Plot types currently supported in Autoplot."""

    #: no plot defined
    empty = auto()

    #: a single 1D line/scatter plot per panel
    singletraces = auto()

    #: multiple 1D lines/scatter plots per panel
    multitraces = auto()

    #: image plot of 2D data
    image = auto()

    #: colormesh plot of 2D data
    colormesh = auto()

    #: 2D scatter plot
    scatter2d = auto()


class SymmetricNorm(colors.Normalize):
    """Color norm that's symmetric and linear around a center value."""

    def __init__(self, vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 vcenter: float = 0,
                 clip: bool = False):
        super().__init__(vmin, vmax, clip)
        self.vcenter = vcenter

    # this only overrides one of the overloaded signatures and therefor fails
    # however on 3.8 this does not fail.
    def __call__(self, value: float, clip: Optional[bool] = None) -> float:  # type: ignore[override,unused-ignore]
        vlim = max(abs(self.vmin - self.vcenter), abs(self.vmax - self.vcenter))
        self.vmax: float = vlim + self.vcenter
        self.vmin: float = -vlim + self.vcenter
        return super().__call__(value, clip)


# 2D plots
def colorplot2d(ax: Axes,
                x: Union[np.ndarray, np.ma.MaskedArray],
                y: Union[np.ndarray, np.ma.MaskedArray],
                z: Union[np.ndarray, np.ma.MaskedArray],
                plotType: PlotType = PlotType.image,
                axLabels: Tuple[Optional[str], Optional[str], Optional[str]] = ('', '', ''),
                **kw: Any) -> Optional[ScalarMappable]:
    """make a 2d colorplot. what plot is made, depends on `plotType`.
    Any of the 2d plot types in :class:`PlotType` works.

    :param ax: matplotlib subPlots to plot in
    :param x: x coordinates (meshgrid)
    :param y: y coordinates (meshgrid)
    :param z: z data
    :param plotType: the plot type
    :param axLabels: labels for the x, y subPlots, and the colorbar.

    all keywords are passed to the actual plotting functions, depending on the ``plotType``:

    - :attr:`PlotType.image` --
        :func:`plotImage`
    - :attr:`PlotType.colormesh` --
        :func:`ppcolormesh_from_meshgrid`
    - :attr:`PlotType.scatter2d` --
        matplotlib's `scatter`
    """
    cmap = kw.pop('cmap', rcParams['image.cmap'])

    # first we need to check if our grid can be plotted nicely.
    if plotType in [PlotType.image, PlotType.colormesh]:
        x = x.astype(float)
        y = y.astype(float)
        z = z.astype(float)

        # first check if we need to fill some masked values in
        if isinstance(x, np.ma.MaskedArray) and np.ma.is_masked(x):
            x = x.filled(np.nan)
        if isinstance(y, np.ma.MaskedArray) and np.ma.is_masked(y):
            y = y.filled(np.nan)
        if isinstance(z, np.ma.MaskedArray) and np.ma.is_masked(z):
            z = z.filled(np.nan)

        # next: try some surgery, if possible
        if np.all(num.is_invalid(x)) or np.all(num.is_invalid(y)):
            return None
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            x, y = interp_meshgrid_2d(x, y)
        if np.any(num.is_invalid(x)) or np.any(num.is_invalid(y)):
            x, y, z = num.crop2d(x, y, z)

        # next, check if the resulting grids are even still plottable
        for g in x, y, z:
            if g.size == 0:
                return None
            elif len(g.shape) < 2:
                return None

            # special case: if we have a single line, a pcolor-type plot won't work.
            elif min(g.shape) < 2:
                plotType = PlotType.scatter2d
    im: Optional[ScalarMappable]
    if plotType is PlotType.image:
        im = plotImage(ax, x, y, z, cmap=cmap, **kw)
    elif plotType is PlotType.colormesh:
        im = ppcolormesh_from_meshgrid(ax, x, y, z, cmap=cmap, **kw)
    elif plotType is PlotType.scatter2d:
        im = ax.scatter(x.ravel(), y.ravel(), c=z.ravel(), cmap=cmap, **kw)
    else:
        im = None

    if im is None:
        return None

    if axLabels[0]:
        ax.set_xlabel(axLabels[0])
    if axLabels[1]:
        ax.set_ylabel(axLabels[1])
    return im


def ppcolormesh_from_meshgrid(ax: Axes, x: np.ndarray, y: np.ndarray,
                              z: np.ndarray, **kw: Any) -> Optional[ScalarMappable]:
    r"""Plot a pcolormesh with some reasonable defaults.
    Input are the corresponding arrays from a 2D ``MeshgridDataDict``.

    Will attempt to fix missing points in the coordinates.

    :param ax: subPlots to plot the colormesh into.
    :param x: x component of the meshgrid coordinates
    :param y: y component of the meshgrid coordinates
    :param z: data values
    :returns: the image returned by `pcolormesh`.

    Keywords are passed on to `pcolormesh`.
    """
    # the meshgrid we have describes coordinates, but for plotting
    # with pcolormesh we need vertices.
    try:
        x = centers2edges_2d(x)
        y = centers2edges_2d(y)
    except:
        return None

    im = ax.pcolormesh(x, y, z, **kw)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return im


def plotImage(ax: Axes, x: np.ndarray, y: np.ndarray,
              z: np.ndarray, **kw: Any) -> AxesImage:
    """Plot 2d meshgrid data as image.

    :param ax: matplotlib subPlots to plot the image in.
    :param x: x coordinates (as meshgrid)
    :param y: y coordinates
    :param z: z values
    :returns: the image object returned by `imshow`

    All keywords are passed to `imshow`.
    """
    ax.grid(False)
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()

    extentx = [x0, x1]
    if x0 > x1:
        extentx = extentx[::-1]
    if x0 == x1:
        extentx = [x0, x0 + 1]
    extenty = [y0, y1]
    if y0 > y1:
        extenty = extenty[::-1]
    if y0 == y1:
        extenty = [y0, y0 + 1]
    extent = cast(Tuple[float,float,float,float], tuple(extentx + extenty))

    if x.shape[0] > 1:
        # in image mode we have to be a little careful:
        # if the x/y subPlots are specified with decreasing values we need to
        # flip the image. otherwise we'll end up with an axis that has the
        # opposite ordering from the data.
        z = z if x[0, 0] < x[1, 0] else z[::-1, :]

    if y.shape[1] > 1:
        z = z if y[0, 0] < y[0, 1] else z[:, ::-1]

    im = ax.imshow(z.T, aspect='auto', origin='lower',
                   extent=extent, **kw)
    return im


# Axis/display helpers -------------------------------------------------------

#: matplotlib axis scales we expose in the GUI.
AXIS_SCALES: Tuple[str, ...] = ('linear', 'log', 'symlog')

#: tick directions we expose in the GUI.
TICK_DIRECTIONS: Tuple[str, ...] = ('out', 'in', 'inout')


def square_axes(ax: Axes) -> None:
    """Give an axes a square plot range with equal aspect ratio.

    The current x and y ranges are replaced by a single, common range
    (the larger of the two spans, centered on the data), and the aspect
    ratio is set to ``'equal'``. This makes, e.g., a circle in the complex
    plane appear as a circle rather than an ellipse.

    :param ax: the axes to modify.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    half = 0.5 * max(abs(x1 - x0), abs(y1 - y0))
    if half == 0 or not np.isfinite(half):
        half = 1.0
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    ax.set_aspect('equal', adjustable='box')


def data_axes(fig: Figure) -> List[Axes]:
    """Return the "data" axes of a figure, i.e. everything except colorbars.

    :param fig: the figure to inspect.
    :return: list of axes that hold plotted data.
    """
    return [ax for ax in fig.axes if ax.get_label() != '<colorbar>']


@dataclass
class AxesOptions:
    """User-adjustable matplotlib axes properties.

    Any attribute left at its "unset" value (``None`` for most, ``False`` for
    :attr:`equalAspect`) leaves the corresponding matplotlib default/automatic
    behaviour untouched. Instances are meant to be re-applied to freshly drawn
    axes so that user choices survive plottr's automatic re-drawing.
    """

    #: x-axis scale, one of :data:`AXIS_SCALES`; ``None`` leaves it as-is.
    xscale: Optional[str] = None
    #: y-axis scale, one of :data:`AXIS_SCALES`; ``None`` leaves it as-is.
    yscale: Optional[str] = None
    #: lower x limit; ``None`` means auto-scale.
    xmin: Optional[float] = None
    #: upper x limit; ``None`` means auto-scale.
    xmax: Optional[float] = None
    #: lower y limit; ``None`` means auto-scale.
    ymin: Optional[float] = None
    #: upper y limit; ``None`` means auto-scale.
    ymax: Optional[float] = None
    #: whether to draw the grid; ``None`` leaves the rcParams default.
    grid: Optional[bool] = None
    #: whether to force an equal (square) aspect ratio.
    equalAspect: bool = False
    #: whether to show minor ticks; ``None`` leaves the default.
    minorTicks: Optional[bool] = None
    #: tick direction, one of :data:`TICK_DIRECTIONS`; ``None`` leaves default.
    tickDirection: Optional[str] = None
    #: whether to draw ticks on the top/right spines as well.
    ticksAllSides: Optional[bool] = None
    #: whether to suppress the shared exponent/offset text on the axes.
    noOffsetText: Optional[bool] = None

    def isDefault(self) -> bool:
        """Whether all options are at their neutral (do-nothing) value."""
        return (
            self.xscale is None and self.yscale is None
            and self.xmin is None and self.xmax is None
            and self.ymin is None and self.ymax is None
            and self.grid is None and not self.equalAspect
            and self.minorTicks is None and self.tickDirection is None
            and self.ticksAllSides is None and self.noOffsetText is None
        )


def apply_axes_options(ax: Axes, options: AxesOptions) -> None:
    """Apply :class:`AxesOptions` to a single matplotlib axes.

    Only options that differ from their neutral value have an effect, so this
    can be called on any axes without clobbering settings the user did not
    explicitly request.

    :param ax: the axes to modify.
    :param options: the options to apply.
    """
    if options.xscale is not None:
        ax.set_xscale(options.xscale)
    if options.yscale is not None:
        ax.set_yscale(options.yscale)

    if options.grid is not None:
        ax.grid(options.grid)

    if options.equalAspect:
        ax.set_aspect('equal', adjustable='box')

    if options.minorTicks is not None:
        if options.minorTicks:
            ax.minorticks_on()
        else:
            ax.minorticks_off()

    tickKw: Dict[str, Any] = {}
    if options.tickDirection is not None:
        tickKw['direction'] = options.tickDirection
    if options.ticksAllSides is not None:
        tickKw.update(top=options.ticksAllSides, right=options.ticksAllSides)
    if tickKw:
        ax.tick_params(which='both', **tickKw)

    if options.noOffsetText:
        # journals generally want the factor spelled out in the axis label
        # rather than as a floating "1e9" in the corner.
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if isinstance(formatter, ScalarFormatter):
                formatter.set_useOffset(False)
                formatter.set_scientific(False)

    # limits are applied last so that they win over any autoscaling triggered
    # by the settings above. A value of None keeps the current (auto) limit.
    if options.xmin is not None or options.xmax is not None:
        ax.set_xlim(left=options.xmin, right=options.xmax)
    if options.ymin is not None or options.ymax is not None:
        ax.set_ylim(bottom=options.ymin, top=options.ymax)


def apply_axes_options_to_figure(fig: Figure, options: AxesOptions) -> None:
    """Apply :class:`AxesOptions` to all data axes of a figure.

    Colorbar axes are left untouched.

    :param fig: the figure whose data axes should be modified.
    :param options: the options to apply.
    """
    if options.isDefault():
        return
    for ax in data_axes(fig):
        apply_axes_options(ax, options)


# Labels, legend and title ---------------------------------------------------

#: legend locations we expose in the GUI.
LEGEND_LOCATIONS: Tuple[str, ...] = (
    'best', 'upper right', 'upper left', 'lower left', 'lower right',
    'right', 'center left', 'center right', 'lower center', 'upper center',
    'center',
)


@dataclass
class LabelOptions:
    """User overrides for the figure's text.

    Labels are normally derived from the data (``name (unit)``), which is fine
    on screen but rarely what a paper wants. Any field left ``None`` keeps the
    automatically derived text.
    """

    #: figure title; ``''`` explicitly removes it, ``None`` keeps the default.
    title: Optional[str] = None
    #: whether to draw the figure title at all.
    showTitle: bool = False
    #: x axis label override.
    xlabel: Optional[str] = None
    #: y axis label override.
    ylabel: Optional[str] = None
    #: colorbar label override.
    colorbarLabel: Optional[str] = None
    #: whether to draw a legend; ``None`` keeps the automatic behaviour.
    showLegend: Optional[bool] = None
    #: legend location, one of :data:`LEGEND_LOCATIONS`.
    legendLocation: Optional[str] = None
    #: whether the legend has a frame.
    legendFrame: Optional[bool] = None
    #: number of legend columns.
    legendColumns: Optional[int] = None

    def isDefault(self) -> bool:
        """Whether all options are at their neutral (do-nothing) value."""
        return (
            self.title is None and not self.showTitle
            and self.xlabel is None and self.ylabel is None
            and self.colorbarLabel is None and self.showLegend is None
            and self.legendLocation is None and self.legendFrame is None
            and self.legendColumns is None
        )


def apply_label_options(fig: Figure, options: LabelOptions) -> None:
    """Apply :class:`LabelOptions` to a figure.

    The x label is applied to every data axes, since panels of a multi-panel
    plot share the same independent variable. The y label only goes to the
    first one, because the panels show different quantities (magnitude and
    phase, say) and one shared label would be wrong.
    """
    if options.isDefault():
        return

    axes = data_axes(fig)
    if not axes:
        return

    if options.xlabel is not None:
        for ax in axes:
            ax.set_xlabel(options.xlabel)
    if options.ylabel is not None:
        axes[0].set_ylabel(options.ylabel)

    if options.colorbarLabel is not None:
        for ax in fig.axes:
            if ax.get_label() == '<colorbar>':
                ax.set_ylabel(options.colorbarLabel)

    if options.showLegend is not None or options.legendLocation is not None \
            or options.legendFrame is not None \
            or options.legendColumns is not None:
        for ax in axes:
            _apply_legend(ax, options)


def _apply_legend(ax: Axes, options: LabelOptions) -> None:
    """Re-draw (or remove) the legend of an axes according to `options`."""
    existing = ax.get_legend()

    if options.showLegend is False:
        if existing is not None:
            existing.remove()
        return

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # only create a legend we weren't asked for if one is already there
    if options.showLegend is None and existing is None:
        return

    kw: Dict[str, Any] = {'fontsize': 'small'}
    kw['loc'] = options.legendLocation or 'best'
    if options.legendFrame is not None:
        kw['frameon'] = options.legendFrame
    if options.legendColumns is not None and options.legendColumns > 0:
        kw['ncol'] = options.legendColumns
    ax.legend(handles, labels, **kw)


def apply_title(fig: Figure, options: LabelOptions, default: str = '') -> None:
    """Set (or clear) the figure title according to `options`.

    :param fig: figure to title.
    :param options: the label options.
    :param default: title to use when the user has not overridden it.
    """
    if not options.showTitle:
        return
    title = options.title if options.title is not None else default
    if title:
        fig.suptitle(title, horizontalalignment='center',
                     verticalalignment='top', fontsize='small')


# Per-trace styling ----------------------------------------------------------

#: line styles we expose in the GUI. ``''`` means no connecting line.
LINE_STYLES: Tuple[str, ...] = ('-', '--', '-.', ':', '')

#: markers we expose in the GUI. ``''`` means no marker.
MARKERS: Tuple[str, ...] = ('', 'o', '.', 's', '^', 'v', 'D', 'x', '+')


@dataclass
class TraceStyle:
    """Appearance overrides for a single plotted line."""

    color: Optional[str] = None
    linestyle: Optional[str] = None
    linewidth: Optional[float] = None
    marker: Optional[str] = None
    markersize: Optional[float] = None
    alpha: Optional[float] = None

    def isDefault(self) -> bool:
        return all(v is None for v in (self.color, self.linestyle,
                                       self.linewidth, self.marker,
                                       self.markersize, self.alpha))


@dataclass
class TraceOptions:
    """Per-trace styling for a figure, keyed by trace index."""

    #: style overrides, keyed by the index of the line within the figure.
    styles: Dict[int, TraceStyle] = field(default_factory=dict)

    #: above this many points, markers are dropped automatically so dense
    #: sweeps do not turn into a solid blob in print. ``None`` disables this.
    markerLimit: Optional[int] = 200

    def isDefault(self) -> bool:
        return (all(s.isDefault() for s in self.styles.values())
                and self.markerLimit is None)


def figure_lines(fig: Figure) -> List[Any]:
    """The data lines of a figure's data axes, in a stable order.

    Helper lines created by ``Axes.errorbar`` (bars and caps) are excluded:
    they are labelled ``_nolegend_`` and are not traces the user styles.
    """
    lines: List[Any] = []
    for ax in data_axes(fig):
        lines.extend(l for l in ax.lines
                     if str(l.get_label()) != '_nolegend_')
    return lines


def apply_trace_options(fig: Figure, options: TraceOptions) -> None:
    """Apply :class:`TraceOptions` to the lines of a figure."""
    if options.isDefault():
        return

    for i, line in enumerate(figure_lines(fig)):
        if options.markerLimit is not None:
            try:
                npoints = len(line.get_xdata())
            except TypeError:
                npoints = 0
            if npoints > options.markerLimit:
                line.set_marker('')

        style = options.styles.get(i)
        if style is None or style.isDefault():
            continue
        if style.color is not None:
            line.set_color(style.color)
        if style.linestyle is not None:
            line.set_linestyle(style.linestyle)
        if style.linewidth is not None:
            line.set_linewidth(style.linewidth)
        if style.marker is not None:
            line.set_marker(style.marker if style.marker else 'None')
        if style.markersize is not None:
            line.set_markersize(style.markersize)
        if style.alpha is not None:
            line.set_alpha(style.alpha)


# Color scale (2D plots) -----------------------------------------------------

@dataclass
class ColorOptions:
    """Color-scale settings for 2D plots."""

    #: name of the colormap; ``None`` keeps the rcParams default.
    colormap: Optional[str] = None
    #: lower limit of the color scale.
    vmin: Optional[float] = None
    #: upper limit of the color scale.
    vmax: Optional[float] = None
    #: whether to use a norm that is symmetric around :attr:`symmetricCenter`.
    symmetric: bool = False
    #: center value for the symmetric norm.
    symmetricCenter: float = 0.0

    def isDefault(self) -> bool:
        return (self.colormap is None and self.vmin is None
                and self.vmax is None and not self.symmetric)


def figure_mappables(fig: Figure) -> List[ScalarMappable]:
    """All color-mapped artists (images, meshes, scatters) of a figure."""
    mappables: List[ScalarMappable] = []
    for ax in data_axes(fig):
        mappables.extend(ax.images)
        mappables.extend(ax.collections)
    return mappables


def apply_color_options(fig: Figure, options: ColorOptions) -> None:
    """Apply :class:`ColorOptions` to the color-mapped artists of a figure."""
    if options.isDefault():
        return

    for mappable in figure_mappables(fig):
        if options.colormap is not None:
            mappable.set_cmap(options.colormap)

        if options.symmetric:
            vmin, vmax = mappable.get_clim()
            if options.vmin is not None:
                vmin = options.vmin
            if options.vmax is not None:
                vmax = options.vmax
            norm = SymmetricNorm(vmin=vmin, vmax=vmax,
                                 vcenter=options.symmetricCenter)
            mappable.set_norm(norm)
        else:
            if options.vmin is not None or options.vmax is not None:
                mappable.set_clim(vmin=options.vmin, vmax=options.vmax)


# Export ---------------------------------------------------------------------

#: named figure widths (in inches) for common journal column layouts.
#: The saved figure must have an exact physical width so that the font sizes
#: come out right on the printed page.
FIGURE_WIDTH_PRESETS: "OrderedDictType[str, Optional[float]]" = OrderedDict([
    ('Auto', None),
    ('APS/IEEE 1-column (3.375 in)', 3.375),
    ('APS/IEEE 2-column (6.9 in)', 6.9),
    ('Nature 1-column (89 mm)', 89.0 / 25.4),
    ('Nature 1.5-column (120 mm)', 120.0 / 25.4),
    ('Nature 2-column (183 mm)', 183.0 / 25.4),
])

#: width used when no preset and no explicit size is given.
DEFAULT_EXPORT_WIDTH = 3.375

#: height/width ratio used to derive a height when none is given.
DEFAULT_EXPORT_ASPECT = 0.77

#: file formats we offer for export.
EXPORT_FORMATS: Tuple[str, ...] = ('pdf', 'svg', 'eps', 'png')

#: rcParams that make text in vector output survive journal submission:
#: fonttype 42 embeds TrueType instead of Type 3 (which many publishers
#: reject), and svg.fonttype 'none' keeps text as text rather than outlines.
FONT_EMBEDDING_RCPARAMS: Dict[str, Any] = {
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}


@dataclass
class ExportSpec:
    """How to render a figure for publication.

    The exported figure is drawn into a *fresh* figure of exactly this size,
    independently of the on-screen window, so the result is reproducible.
    """

    #: figure width in inches; ``None`` means :data:`DEFAULT_EXPORT_WIDTH`.
    width: Optional[float] = None
    #: figure height in inches; ``None`` derives it from the width.
    height: Optional[float] = None
    #: resolution for raster output.
    dpi: int = 300
    #: output file format, one of :data:`EXPORT_FORMATS`.
    format: str = 'pdf'
    #: whether the background is transparent.
    transparent: bool = False
    #: base font size in points for the exported figure. Deliberately
    #: independent of the screen's DPI scaling.
    fontSize: float = 8.0
    #: font family for the exported figure; ``None`` keeps the current one.
    fontFamily: Optional[str] = None

    def size(self) -> Tuple[float, float]:
        """Resolve the figure size in inches."""
        width = self.width if self.width is not None else DEFAULT_EXPORT_WIDTH
        height = self.height if self.height is not None \
            else width * DEFAULT_EXPORT_ASPECT
        return width, height

    def rcParams(self) -> Dict[str, Any]:
        """rcParams to render the exported figure under."""
        params: Dict[str, Any] = dict(FONT_EMBEDDING_RCPARAMS)
        params['font.size'] = self.fontSize
        params['savefig.dpi'] = self.dpi
        params['figure.dpi'] = self.dpi
        if self.fontFamily:
            params['font.family'] = self.fontFamily
        return params


def save_figure(fig: Figure, filepath: str, spec: ExportSpec) -> None:
    """Write `fig` to `filepath` according to `spec`.

    Note that ``bbox_inches='tight'`` is deliberately *not* used: it changes
    the final size (a 3.375 in figure comes out 3.49 in), which defeats the
    purpose of specifying a column width. Spacing is handled by the figure's
    constrained layout instead.
    """
    fig.savefig(filepath, dpi=spec.dpi, transparent=spec.transparent,
                facecolor='none' if spec.transparent else 'white')
